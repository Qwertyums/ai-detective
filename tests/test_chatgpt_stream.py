import json
from unittest.mock import patch
import httpx
import pytest
from openai import OpenAI
from src.chatgpt_auth import AuthError, RESOURCE
from src.chatgpt_inference import infer_chatgpt, list_models, _catalog
from src.data_validation import load_dataset
from src.llm import LLMError, configuration


MODEL='gpt-6-luna'


def output():
    return {'question_id':'Q1','status':'entailed','answer':'10:10','evidence_ids':['C21'],
            'counterevidence_ids':[],'confidence':.9,'explanation':'Счёт указан в C21.','missing_information':None}


def terminal(status='completed',code=None):
    response={'id':'resp_plan','object':'response','created_at':1,'model':MODEL,'status':status,
              'output':[{'id':'msg_plan','type':'message','role':'assistant','status':'completed',
                         'content':[{'type':'output_text','text':json.dumps(output()),'annotations':[]}]}],
              'usage':{'input_tokens':120,'output_tokens':50,'total_tokens':170,
                       'input_tokens_details':{'cached_tokens':10},'output_tokens_details':{'reasoning_tokens':3}}}
    if code: response['error']={'code':code,'message':'must-not-copy-server-message'}
    return {'type':'response.'+status,'sequence_number':2,'response':response}


def sse(events):
    return ''.join('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n' for event in events).encode()


class FakeAuth:
    def __init__(self,profile='plan-profile',token='subscription-token'):
        self.profile,self.token=profile,token
        self.requests=[]
    def access(self): return self.token,self.profile
    def _json(self,method,url,**kwargs):
        self.requests.append((method,url,kwargs))
        return {'models':[{'slug':MODEL,'display_name':'Luna','visibility':'list'},
                          {'slug':'hidden-model','display_name':'Hidden','visibility':'hide'}]}


def invoke(events,auth=None,expected_profile=None,status=200,body=None):
    requests=[]
    def handle(request):
        requests.append(request)
        if status!=200:
            return httpx.Response(status,json=body,headers={'x-request-id':'req_plan'})
        return httpx.Response(200,content=sse(events),headers={'content-type':'text/event-stream','x-request-id':'req_plan'})
    client=OpenAI(api_key='subscription-token',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    _catalog.clear()
    with patch('src.chatgpt_inference.OpenAI',return_value=client) as factory:
        result=infer_chatgpt(load_dataset()[2][0],'C21: счёт 10:10',['C21'],MODEL,expected_profile,auth or FakeAuth())
    return result,requests,factory


def test_subscription_stream_uses_oauth_no_unsupported_fields_and_server_usage(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','paid-key-must-not-be-used')
    event={'type':'response.output_text.delta','sequence_number':1,'item_id':'msg_plan',
           'output_index':0,'content_index':0,'delta':json.dumps(output())}
    result,requests,factory=invoke([event,terminal()])
    request=json.loads(requests[0].content)
    assert requests[0].headers['authorization']=='Bearer subscription-token'
    assert factory.call_args.kwargs['api_key']=='subscription-token'
    assert factory.call_args.kwargs['base_url']==RESOURCE
    assert factory.call_args.kwargs['max_retries']==0
    assert request['stream'] is True and request['store'] is False
    assert request['instructions'] and all(m['role']!='system' for m in request['input'])
    assert not {'max_output_tokens','truncation','temperature','top_p','previous_response_id','conversation','metadata'} & set(request)
    assert request['text']['format']['strict'] is True
    assert result.output.answer=='10:10' and result.usage['input_tokens']==120
    assert result.metadata['auth_profile']=='plan-profile' and result.metadata['request_id']=='req_plan'
    assert 'subscription-token' not in json.dumps(result.metadata)
    assert len(requests)==1


@pytest.mark.parametrize('kind',['failed','incomplete'])
def test_terminal_error_is_not_a_success_or_insufficient(kind):
    event=terminal(kind,'subscription_sharing_usage_limit_exceeded' if kind=='failed' else None)
    with pytest.raises(LLMError) as failure: invoke([event])
    assert failure.value.metadata['status']==kind
    assert failure.value.usage['input_tokens']==120
    if kind=='failed':
        assert 'лимит подписки' in str(failure.value)
        assert failure.value.metadata['error_code']=='subscription_sharing_usage_limit_exceeded'
    assert 'must-not-copy-server-message' not in str(failure.value)


def test_stream_end_without_completed_is_failure():
    event={'type':'response.output_text.delta','sequence_number':1,'item_id':'msg_plan',
           'output_index':0,'content_index':0,'delta':json.dumps(output())}
    with pytest.raises(LLMError,match='без response.completed') as failure: invoke([event])
    assert failure.value.raw_response==json.dumps(output())


def test_usage_limit_http_error_does_not_fall_back_to_api_key():
    with pytest.raises(LLMError,match='лимит подписки') as failure:
        invoke([],status=429,body={'error':{'type':'rate_limit_error',
              'code':'subscription_sharing_usage_limit_exceeded','message':'do-not-copy'}})
    assert failure.value.metadata['error_code']=='subscription_sharing_usage_limit_exceeded'
    assert len(failure.value.attempts)==1


def test_model_discovery_is_profile_specific_and_filters_hidden_models():
    _catalog.clear()
    a,b=FakeAuth('account-a'),FakeAuth('account-b')
    assert list_models(a)==[{'slug':MODEL,'display_name':'Luna'}]
    list_models(a)
    list_models(b)
    assert len(a.requests)==1 and len(b.requests)==1
    assert a.requests[0][1]==RESOURCE+'/models'


def test_account_switch_during_request_setup_is_rejected():
    with pytest.raises(LLMError,match='Аккаунт ChatGPT изменился'):
        invoke([],auth=FakeAuth('new-account'),expected_profile='original-account')


def test_model_not_in_catalog_is_rejected_before_inference():
    _catalog.clear()
    with patch('src.chatgpt_inference.OpenAI') as sdk:
        with pytest.raises(LLMError,match='Модель недоступна'):
            infer_chatgpt(load_dataset()[2][0],'context',['C21'],'unavailable-model',auth=FakeAuth())
    sdk.assert_not_called()


def test_configuration_keeps_plan_and_api_experiments_separate():
    _,plan=configuration('chatgpt',MODEL,include_profile=False)
    _,paid=configuration('api_key',MODEL)
    assert plan['provider']=='chatgpt' and plan['stream'] is True
    assert 'max_output_tokens' not in plan and 'truncation' not in plan
    assert paid['provider']=='openai' and 'max_output_tokens' in paid
