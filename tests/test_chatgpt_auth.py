import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from src import chatgpt_auth as module
from src.chatgpt_auth import (AuthError, AuthStore, ChatGPTAuth, ISSUER, TOKEN, JWKS, DIRECT_SCOPE)


@pytest.fixture(scope='module')
def signing():
    key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid='test-key',use='sig',alg='RS256')
    return key,jwk


@pytest.fixture
def auth_env(tmp_path,monkeypatch,signing):
    # A test-only codec keeps unit tests independent of the host Windows account.
    # Production never falls back from DPAPI to this codec.
    def codec(data,decrypt=False):
        return base64.b64decode(data[5:]) if decrypt else b'test:'+base64.b64encode(data)
    monkeypatch.setattr(module,'_dpapi',codec)
    calls, replies = [], {}
    def handle(request):
        calls.append(request)
        if str(request.url) == JWKS:
            return httpx.Response(200,json={'keys':[signing[1]]})
        result = replies.get((request.method,str(request.url)), {'unexpected':True})
        if callable(result):
            return result(request)
        if isinstance(result,httpx.Response):
            return result
        return httpx.Response(200,json=result)
    client = httpx.Client(transport=httpx.MockTransport(handle))
    store = AuthStore(tmp_path/'auth')
    auth = ChatGPTAuth(store,client)
    yield auth,calls,replies
    client.close()


def id_token(signing,client_id,nonce=None,subject='alice',**updates):
    claims = {'iss':ISSUER,'sub':subject,'aud':client_id,'iat':int(time.time()),
              'exp':int(time.time())+3600,'email':'same@example.test',**updates}
    if nonce is not None:
        claims['nonce'] = nonce
    return jwt.encode(claims,signing[0],algorithm='RS256',headers={'kid':'test-key'})


def grant(signing,cid,nonce=None,subject='alice',scope=None,**updates):
    return {'access_token':'access-'+subject,'refresh_token':'refresh-'+subject,
            'id_token':id_token(signing,cid,nonce,subject),'token_type':'Bearer','expires_in':3600,
            'scope':scope if scope is not None else 'openid offline_access resource.invoke '+DIRECT_SCOPE,**updates}


def seed(auth,signing,cid='oaiapp_alice',subject='alice',expires=None):
    with auth.store.lock():
        state=auth.store.read()
        state['profiles'][cid]={'client_id':cid,'issuer':ISSUER,'subject':subject,'email':'same@example.test',
                               'scopes':[DIRECT_SCOPE],'access_token':'access-'+subject,'refresh_token':'refresh-'+subject,
                               'id_token':id_token(signing,cid,subject=subject),'expires_at':expires or time.time()+3600}
        state['active']=cid
        auth.store.write(state)
    return cid


def callback(flow,**query):
    response=httpx.get(flow.redirect_uri+'?'+urlencode({'state':flow.state,'code':'fake-code',**query}),trust_env=False,timeout=3)
    assert response.status_code==200
    assert flow.received.wait(2)


def test_dynamic_pkce_registration_and_reusable_host(auth_env):
    auth,_,_=auth_env
    flow=auth.begin()
    try:
        params=parse_qs(urlsplit(flow.url).query)
        assert params['client_id']==['dynamic_agent_client']
        assert params['agent_name_hint']==['AI Detective']
        assert params['ext_agent_host_id']==[auth.host_id()]
        assert params['redirect_uri']==[flow.redirect_uri]
        assert urlsplit(flow.redirect_uri).hostname=='127.0.0.1'
        assert params['code_challenge']==[base64.urlsafe_b64encode(hashlib.sha256(flow.verifier.encode()).digest()).rstrip(b'=').decode()]
        assert DIRECT_SCOPE in params['scope'][0]
        assert 'client_secret' not in params and 'id_token_hint' not in params
    finally:
        flow.close()
    assert ChatGPTAuth(auth.store).host_id()==auth.host_id()


@pytest.mark.parametrize('bad_state',['incorrect','неверный',''])
def test_callback_rejects_wrong_state_without_consuming_login(auth_env,bad_state):
    auth,calls,_=auth_env
    flow=auth.begin()
    try:
        result=httpx.get(flow.redirect_uri+'?'+urlencode({'state':bad_state,'code':'fake'}),trust_env=False,timeout=3)
        assert result.status_code==400 and not flow.received.is_set()
        assert not calls
    finally:
        flow.close()


def test_successful_login_validates_id_token_and_uses_issued_id(auth_env,signing):
    auth,calls,replies=auth_env
    flow=auth.begin()
    cid='oaiapp_issued'
    replies[('POST',TOKEN)]=grant(signing,cid,flow.nonce)
    verifier=flow.verifier
    try:
        callback(flow,client_id=cid)
        result=flow.finish()
        assert result['sharing'] and result['profile_id']==cid
        data=parse_qs(next(r for r in calls if str(r.url)==TOKEN).content.decode())
        assert data['client_id']==[cid] and data['code_verifier']==[verifier]
        assert data['redirect_uri']==[flow.redirect_uri]
        assert not flow.verifier and flow.callback is None
        public=json.dumps(auth.profiles())
        assert 'access-alice' not in public and 'refresh-alice' not in public
        assert auth.access()[0]=='access-alice'
        assert auth.profiles()['active']==cid
        with pytest.raises(AuthError,match='уже завершена'):
            flow.finish()
    finally:
        flow.close()


def test_identity_only_login_does_not_authorize_inference(auth_env,signing):
    auth,_,replies=auth_env
    flow=auth.begin()
    replies[('POST',TOKEN)]=grant(signing,'oaiapp_identity',flow.nonce,scope='openid profile email')
    try:
        callback(flow,client_id='oaiapp_identity')
        assert not flow.finish()['sharing']
        with pytest.raises(AuthError,match='без разрешения'):
            auth.access()
    finally:
        flow.close()


@pytest.mark.parametrize('case',['nonce','audience','issuer','expired','subject','signature'])
def test_id_token_validation_rejects_invalid_identity(auth_env,signing,case):
    auth,_,_=auth_env
    changes={}
    if case=='audience': changes['aud']='other-client'
    if case=='issuer': changes['iss']='https://untrusted.invalid'
    if case=='expired': changes['exp']=int(time.time())-100
    token=id_token(signing,'oaiapp_test','wrong' if case=='nonce' else 'expected',**changes)
    if case=='signature':
        parts=token.split('.')
        parts[2]=('A' if parts[2][0]!='A' else 'B')+parts[2][1:]
        token='.'.join(parts)
    expected={'issuer':ISSUER,'subject':'bob'} if case=='subject' else None
    with pytest.raises(AuthError,match='проверить личность'):
        auth.validate_identity(token,'oaiapp_test','expected',expected)


def test_reauthorization_cannot_replace_another_account(auth_env,signing):
    auth,_,replies=auth_env
    cid=seed(auth,signing)
    flow=auth.begin(cid)
    params=parse_qs(urlsplit(flow.url).query)
    assert params['client_id']==[cid] and 'agent_name_hint' not in params
    replies[('POST',TOKEN)]=grant(signing,cid,flow.nonce,subject='bob')
    try:
        callback(flow,client_id=cid)
        with pytest.raises(AuthError): flow.finish()
        assert auth.access()[0]=='access-alice'
    finally:
        flow.close()


def test_cancelled_login_does_not_exchange_code_or_change_active_account(auth_env,signing):
    auth,calls,_=auth_env
    active=seed(auth,signing)
    flow=auth.begin()
    try:
        callback(flow,error='access_denied')
        with pytest.raises(AuthError,match='отменены'): flow.finish()
        assert not calls and auth.profiles()['active']==active
    finally:
        flow.close()


def test_missing_issued_client_id_is_not_used_for_exchange(auth_env):
    auth,calls,_=auth_env
    flow=auth.begin()
    try:
        callback(flow)
        with pytest.raises(AuthError,match='client_id'): flow.finish()
        assert not calls
    finally:
        flow.close()


def test_failed_exchange_retains_registration_for_retry(auth_env):
    auth,_,replies=auth_env
    flow=auth.begin()
    replies[('POST',TOKEN)]=httpx.Response(400,json={'error':'invalid_grant'})
    try:
        callback(flow,client_id='oaiapp_pending')
        with pytest.raises(AuthError): flow.finish()
        assert auth.profiles()['active'] is None
        retry=auth.begin('oaiapp_pending')
        try:
            assert parse_qs(urlsplit(retry.url).query)['client_id']==['oaiapp_pending']
        finally: retry.close()
    finally: flow.close()


def test_rotating_refresh_is_serialized_and_saved(auth_env,signing):
    auth,calls,replies=auth_env
    cid=seed(auth,signing,expires=time.time()-1)
    replies[('POST',TOKEN)]={'access_token':'renewed','refresh_token':'replacement',
                             'token_type':'Bearer','expires_in':3600}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:auth.access()[0],range(2)))
    assert results==['renewed','renewed']
    token_calls=[r for r in calls if str(r.url)==TOKEN]
    assert len(token_calls)==1
    data=parse_qs(token_calls[0].content.decode())
    assert data['client_id']==[cid] and data['grant_type']==['refresh_token'] and 'scope' not in data
    with auth.store.lock():
        assert auth.store.read()['profiles'][cid]['refresh_token']=='replacement'


@pytest.mark.parametrize('terminal',[True,False])
def test_refresh_error_clears_only_confirmed_invalid_tokens(auth_env,signing,terminal):
    auth,_,replies=auth_env
    cid=seed(auth,signing,expires=time.time()-1)
    replies[('POST',TOKEN)]=httpx.Response(400 if terminal else 503,json={'error':'invalid_grant' if terminal else 'temporary'})
    with pytest.raises(AuthError): auth.access()
    with auth.store.lock():
        profile=auth.store.read()['profiles'][cid]
        assert bool(profile.get('refresh_token')) is (not terminal)
        assert profile['client_id']==cid and profile['subject']=='alice'


def test_logout_revokes_and_retains_mapping_and_host(auth_env,signing):
    auth,calls,replies=auth_env
    cid=seed(auth,signing)
    host=auth.host_id()
    endpoint=ISSUER+'/api/accounts/oauth/revoke'
    replies[('GET',module.DISCOVERY)]={'issuer':ISSUER,'revocation_endpoint':endpoint}
    replies[('POST',endpoint)]=httpx.Response(200)
    assert auth.logout()
    assert auth.host_id()==host
    with auth.store.lock():
        profile=auth.store.read()['profiles'][cid]
        assert not profile.get('access_token') and not profile.get('refresh_token')
        assert profile['subject']=='alice'
    data=parse_qs(next(r for r in calls if str(r.url)==endpoint).content.decode())
    assert data['token_type_hint']==['refresh_token'] and data['client_id']==[cid]


def test_same_email_does_not_merge_profiles(auth_env,signing):
    auth,_,_=auth_env
    a=seed(auth,signing,cid='oaiapp_a',subject='alice')
    b=seed(auth,signing,cid='oaiapp_b',subject='bob')
    info=auth.profiles()
    assert len(info['profiles'])==2 and info['profiles'][0]['label']!=info['profiles'][1]['label']
    auth.select(a)
    assert auth.access()[0]=='access-alice'
    auth.select(b)
    assert auth.access()[0]=='access-bob'


@pytest.mark.parametrize('updates',[{'token_type':None},{'scope':None},{'expires_in':0},{'expires_in':float('inf')}])
def test_malformed_token_response_cannot_grant_access(auth_env,updates):
    auth,_,_=auth_env
    response={'token_type':'Bearer','scope':DIRECT_SCOPE,'access_token':'fake','expires_in':3600,**updates}
    with pytest.raises(AuthError):
        auth._credentials(response,{'scopes':[DIRECT_SCOPE]})


@pytest.mark.skipif(os.name!='nt',reason='Windows DPAPI')
def test_real_dpapi_encrypts_and_roundtrips_credentials(tmp_path):
    store=AuthStore(tmp_path/'encrypted')
    state={'version':1,'host_id':'urn:uuid:test','active':None,'profiles':{'test':{'access_token':'fake-sensitive-access'}}}
    with store.lock():
        store.write(state)
        assert b'fake-sensitive-access' not in store.path.read_bytes()
        assert store.read()==state
