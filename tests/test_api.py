import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import httpx
import pytest
from openai import OpenAI, AuthenticationError, APITimeoutError
from src.data_validation import load_dataset
from src.llm import SETTINGS, MODEL_NAME, LLMError, create_client, infer, output_schema
from src.experiments import run_one


def payload():
    return {'question_id':'Q1','status':'entailed','answer':'10:10','evidence_ids':['C21'],
            'counterevidence_ids':[],'confidence':.9,'explanation':'Счёт указан в C21.',
            'missing_information':None}


def response_body(output=None, status='completed'):
    return {'id':'resp_test','object':'response','created_at':1,'model':MODEL_NAME,
            'status':status, 'output': output if output is not None else [
                {'id':'rs_test','type':'reasoning','summary':[],'content':None},
                {'id':'msg_test','type':'message','role':'assistant','status':'completed',
                 'content':[{'type':'output_text','text':json.dumps(payload()),'annotations':[]}]}],
            'usage':{'input_tokens':100,'output_tokens':40,'total_tokens':140,
                     'input_tokens_details':{'cached_tokens':20},'output_tokens_details':{'reasoning_tokens':5}}}


def test_strict_schema_has_all_fields_and_context_specific_ids():
    schema = output_schema('Q1',['C21'])
    assert set(schema['required']) == set(schema['properties'])
    assert schema['additionalProperties'] is False
    assert schema['properties']['question_id']['enum'] == ['Q1']
    assert schema['properties']['evidence_ids']['items']['enum'] == ['C21']
    assert all('default' not in p for p in schema['properties'].values())


def test_actual_sdk_sends_structured_request_and_reads_server_usage():
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,json=response_body(),headers={'x-request-id':'request_test'})
    # The real SDK runs against a local mock transport. No paid/network request.
    client = OpenAI(api_key='test-only',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    with patch('src.llm.create_client',return_value=client):
        result = infer(load_dataset()[2][0], 'C21: счёт 10:10', ['C21'])
    request = requests[0]
    assert len(requests) == 1 and request['model'] == MODEL_NAME
    assert request['store'] is False and request['truncation'] == 'disabled'
    assert request['text']['format']['strict'] is True
    assert request['text']['format']['schema']['properties']['evidence_ids']['items']['enum'] == ['C21']
    assert 'C21' in request['input'][1]['content']
    assert 'temperature' not in request and 'seed' not in request
    assert result.output.answer == '10:10' and result.usage['input_tokens'] == 100
    assert result.usage['cached_input_tokens'] == 20 and result.usage['reasoning_tokens'] == 5
    assert result.metadata['request_id'] == 'request_test'
    assert result.model_revision == MODEL_NAME


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with patch('src.llm.OpenAI') as sdk:
        with pytest.raises(LLMError,match='OPENAI_API_KEY'):
            create_client()
    sdk.assert_not_called()


def test_client_uses_official_endpoint_and_no_hidden_retries(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    with patch('src.llm.OpenAI') as sdk:
        create_client()
    assert sdk.call_args.kwargs['max_retries'] == 0
    assert sdk.call_args.kwargs['base_url'] == 'https://api.openai.com/v1'


@pytest.mark.parametrize('kind',['refusal','incomplete'])
def test_api_refusal_and_truncation_are_errors_not_insufficient(kind):
    body = response_body(status='incomplete' if kind == 'incomplete' else 'completed')
    if kind == 'refusal':
        body['output'][-1]['content'] = [{'type':'refusal','refusal':'No answer'}]
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200,json=body)
    client = OpenAI(api_key='test-only',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    with patch('src.llm.create_client',return_value=client):
        with pytest.raises(LLMError) as failure:
            infer(load_dataset()[2][0], 'context', ['C21'])
    assert len(calls) == 1
    assert failure.value.usage['input_tokens'] == 100
    assert failure.value.metadata['status'] == body['status']


@pytest.mark.parametrize('kind',['auth','timeout'])
def test_api_errors_are_sanitized_and_not_retried(kind):
    request = httpx.Request('POST','https://api.openai.com/v1/responses')
    error = (AuthenticationError('secret-key-must-not-appear',response=httpx.Response(401,request=request),body=None)
             if kind == 'auth' else APITimeoutError(request=request))
    client = Mock()
    client.responses.create.side_effect = error
    with patch('src.llm.create_client',return_value=client):
        with pytest.raises(LLMError) as failure:
            infer(load_dataset()[2][0], 'context', ['C21'])
    assert 'secret-key-must-not-appear' not in str(failure.value)
    assert client.responses.create.call_count == 1
    client.close.assert_called_once()


@pytest.mark.parametrize('code,expected', [
    ('insufficient_quota', 'Недостаточно квоты'),
    ('credit_balance_exhausted', 'предоплаченные кредиты'),
    ('project_spend_limit_exceeded', 'расходов API-проекта'),
    ('organization_spend_limit_exceeded', 'расходов организации'),
    ('organization_usage_limit_exceeded', 'квота использования организации'),
    ('usage_limit_exceeded', 'лимит использования API'),
    ('rate_limit_exceeded', 'Временно превышен'),
    ('slow_down', 'выросла слишком быстро'),
])
def test_429_diagnostics_are_persisted_without_retry_or_secret(code, expected, tmp_path):
    calls = []
    def handle(request):
        calls.append(request)
        # A more specific code must take precedence over the broad quota type.
        return httpx.Response(429, json={'error':{'code':code,'type':'insufficient_quota',
                                                  'message':'secret-key-must-not-appear'}},
                              headers={'retry-after':'30','x-request-id':'req_test429'})
    client = OpenAI(api_key='test-only',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    docs, claims, questions = load_dataset()
    with patch('src.llm.create_client',return_value=client):
        record = run_one(questions[0],'B',docs,claims,output_dir=tmp_path,provider='api_key')
    assert expected in record['error']
    assert record['output'] is None and record['evaluation'] is None
    assert len(calls) == 1
    assert record['api_metadata']['error_code'] == code
    assert record['api_metadata']['request_id'] == 'req_test429'
    assert record['api_metadata']['retry_after_seconds'] == 30
    assert 'secret-key-must-not-appear' not in json.dumps(record)
    assert ('не раньше чем через 30 с' in record['error']) == (code in {'rate_limit_exceeded','slow_down'})
    saved = json.loads(next(tmp_path.glob('*.json')).read_text(encoding='utf-8'))
    assert saved['api_metadata'] == record['api_metadata']


def test_unknown_429_does_not_claim_balance_is_exhausted():
    from src.llm import rate_limit_message, api_error_details
    from openai import RateLimitError
    exc = RateLimitError('secret-key-must-not-appear', response=httpx.Response(429,request=httpx.Request('POST','https://api.openai.com/v1/responses')),
                         body={'error':{'code':'secret-key-must-not-appear','type':'unknown'}})
    details = api_error_details(exc)
    assert details['error_code'] is None and details['error_type'] is None
    assert 'может быть' in rate_limit_message(details)
    assert 'secret-key' not in json.dumps(details)


@pytest.mark.parametrize('header',['invalid','NaN','inf','-5'])
def test_invalid_retry_after_is_not_displayed(header):
    from src.llm import api_error_details
    from openai import RateLimitError
    exc = RateLimitError('test',response=httpx.Response(429,headers={'retry-after':header},request=httpx.Request('POST','https://api.openai.com/v1/responses')),
                         body={'error':{'code':'rate_limit_exceeded','type':'rate_limit_error'}})
    details = api_error_details(exc)
    assert details['error_code'] == 'rate_limit_exceeded'
    assert 'retry_after_seconds' not in details
