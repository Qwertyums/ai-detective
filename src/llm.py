"""OpenAI Responses API with strict structured output and local evidence checks."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError
from src.schemas import ModelOutput

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env', override=False)
MODEL_NAME = os.environ.get('OPENAI_MODEL', 'gpt-6-luna').strip() or 'gpt-6-luna'
DEFAULT_PROVIDER = os.environ.get('OPENAI_AUTH_MODE', 'chatgpt').strip() or 'chatgpt'
SETTINGS = {'provider': 'openai', 'api': 'responses', 'max_output_tokens': 2048,
            'reasoning_effort': 'low' if MODEL_NAME.startswith(('gpt-5', 'gpt-6')) else None,
            'temperature': None, 'store': False, 'truncation': 'disabled',
            'timeout_seconds': 60, 'sdk_max_retries': 0,
            'schema_retries': 1, 'output_contract_version': 5}

class LLMError(RuntimeError):
    def __init__(self, message, raw_response=None, usage=None, attempts=None, metadata=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.usage = usage or {}
        self.attempts = attempts or []
        self.metadata = metadata or {}

@dataclass
class Inference:
    output: ModelOutput
    raw_response: str
    usage: dict
    model_revision: str | None
    attempts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def configuration(provider=None, model=None, include_profile=True):
    provider, model = provider or DEFAULT_PROVIDER, model or MODEL_NAME
    if provider not in {'chatgpt', 'api_key'}:
        raise LLMError('OPENAI_AUTH_MODE должен быть chatgpt или api_key.')
    settings = dict(SETTINGS)
    settings['reasoning_effort'] = 'low' if model.startswith(('gpt-5', 'gpt-6')) else None
    if provider == 'chatgpt':
        for key in ('max_output_tokens', 'temperature', 'truncation'):
            settings.pop(key, None)
        settings.update(provider='chatgpt', stream=True, output_contract_version=6)
        settings['auth_profile'] = None
        if include_profile:
            from src.chatgpt_auth import ChatGPTAuth, AuthError
            try:
                accounts = ChatGPTAuth().profiles()
                active = next((p for p in accounts['profiles'] if p['profile_id'] == accounts['active']), None)
                settings['auth_profile'] = active.get('auth_profile') if active else None
            except AuthError as exc:
                raise LLMError(str(exc)) from None
    else:
        settings['provider'] = 'openai'
    return model, settings

def api_configured():
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    return bool(key and key != 'your_api_key_here')

def create_client():
    if not api_configured():
        raise LLMError('Не задан OPENAI_API_KEY. Скопируйте .env.example в .env и добавьте свой API-ключ, затем перезапустите приложение.')
    # Pin the official endpoint, independent of unrelated machine-wide settings.
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'].strip(), base_url='https://api.openai.com/v1',
                  timeout=SETTINGS['timeout_seconds'], max_retries=SETTINGS['sdk_max_retries'])

def system_prompt():
    return (ROOT / 'prompts/system_prompt.txt').read_text(encoding='utf-8-sig')

def make_messages(question, context, allowed_ids=None):
    return [{'role':'system', 'content':system_prompt()},
            {'role':'user', 'content':f'Досье:\n{context}\n\nquestion_id: {question.question_id}\nВопрос: {question.question}\n'
             'Ответь по заданной JSON-схеме. Не пропускай обязательные поля. '
             'Используй только доступные основания и краткое проверяемое объяснение.'
             + ('\nДопустимые claim_id: ' + ', '.join(allowed_ids) if allowed_ids is not None else '')}]

def output_schema(question_id, allowed_ids):
    schema = ModelOutput.model_json_schema()
    # Strict Structured Outputs requires every property, including nullable ones.
    schema['required'] = list(schema['properties'])
    schema['additionalProperties'] = False
    for prop in schema['properties'].values():
        prop.pop('default', None)
    schema['properties']['question_id']['enum'] = [question_id]
    for key in ('evidence_ids', 'counterevidence_ids'):
        schema['properties'][key]['items'] = {'type':'string', 'enum':sorted(set(allowed_ids))}
    return schema

def normalize_response(raw):
    """Legacy syntax normalization, with no invented field values."""
    text = raw.strip()
    changes = []
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
        changes.append('markdown_fence_removed')
    try:
        json.loads(text)
    except json.JSONDecodeError:
        names = '|'.join(re.escape(name) for name in ModelOutput.model_fields)
        pattern = r'"(?:\\.|[^"\\])*"|(?P<prefix>[{,]\s*)(?P<key>' + names + r')(?P<colon>\s*:)'
        def quote_key(match):
            if match.group('key') is None:
                return match.group(0)
            return match.group('prefix') + json.dumps(match.group('key')) + match.group('colon')
        normalized = re.sub(pattern, quote_key, text)
        if normalized != text:
            changes.append('quoted_property_names')
            text = normalized
    return text, changes

def parse_output(raw, question_id, allowed_ids):
    try:
        text, _ = normalize_response(raw)
        data = json.loads(text)
        if isinstance(data, dict):
            data.setdefault('question_id', question_id)
        return ModelOutput.model_validate(data, context={'question_id':question_id, 'allowed_ids':allowed_ids})
    except ValidationError as exc:
        details = exc.errors(include_url=False, include_input=False)
        missing = ['.'.join(map(str,e['loc'])) for e in details if e['type']=='missing']
        other = [('.'.join(map(str,e['loc'])) or 'ответ')+': '+e['msg'] for e in details if e['type']!='missing']
        message = ('Отсутствуют обязательные поля: '+', '.join(missing)+'. ') if missing else ''
        raise LLMError('Невалидный ответ LLM: '+(message+'; '.join(other)).strip(), raw_response=raw) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LLMError('Невалидный ответ LLM: '+str(exc), raw_response=raw) from exc


def api_error_details(exc):
    """Keep diagnostic categories, never raw server messages or request headers."""
    body = getattr(exc, 'body', None)
    if not isinstance(body, dict):
        body = {}
    if isinstance(body.get('error'), dict):
        body = body['error']
    codes = {
        'insufficient_quota', 'credit_balance_exhausted', 'organization_spend_limit_exceeded',
        'project_spend_limit_exceeded', 'organization_usage_limit_exceeded', 'usage_limit_exceeded',
        'rate_limit_exceeded', 'slow_down',
        'subscription_sharing_usage_limit_exceeded', 'subscription_sharing_usage_unavailable',
        'subscription_sharing_unsupported_capability', 'subscription_sharing_route_not_supported',
        'subscription_sharing_invalid_user', 'chatpass_v2_scope_not_authorized',
        'chatpass_v2_invalid_authorization_context', 'subscription_sharing_user_unavailable',
    }
    types = {'insufficient_quota', 'rate_limit_error', 'tokens', 'requests'}
    code = body.get('code') or getattr(exc, 'code', None)
    error_type = body.get('type') or getattr(exc, 'type', None)
    details = {'http_status': getattr(exc, 'status_code', None),
               'error_code': code if isinstance(code, str) and code in codes else None,
               'error_type': error_type if isinstance(error_type, str) and error_type in types else None}
    response = getattr(exc, 'response', None)
    if response is not None:
        request_id = response.headers.get('x-request-id')
        if request_id and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
            details['request_id'] = request_id
        try:
            seconds = float(response.headers.get('retry-after', ''))
            if math.isfinite(seconds) and seconds >= 0:
                details['retry_after_seconds'] = seconds
        except (TypeError, ValueError):
            pass
    return details


def rate_limit_message(details):
    code = details['error_code']
    messages = {
        'credit_balance_exhausted': 'Закончились предоплаченные кредиты OpenAI API. Проверьте Billing организации.',
        'insufficient_quota': 'Недостаточно квоты OpenAI API. Проверьте баланс и доступную квоту организации/проекта в Billing и Limits.',
        'organization_spend_limit_exceeded': 'Достигнут лимит расходов организации. Проверьте Limits организации; ожидание между запросами не устраняет этот лимит.',
        'project_spend_limit_exceeded': 'Достигнут лимит расходов API-проекта. Проверьте лимит выбранного проекта; ожидание между запросами не устраняет этот лимит.',
        'organization_usage_limit_exceeded': 'Достигнута квота использования организации, установленная OpenAI. Проверьте Limits организации.',
        'usage_limit_exceeded': 'Достигнут лимит использования API. Проверьте Billing и Limits организации/проекта.',
        'rate_limit_exceeded': 'Временно превышен лимит запросов или токенов. Уменьшите частоту запросов и учитывайте Retry-After.',
        'slow_down': 'Частота запросов выросла слишком быстро. Уменьшите её и учитывайте Retry-After.',
    }
    if code in messages:
        reason = messages[code]
    elif details['error_type'] == 'insufficient_quota':
        reason = messages['insufficient_quota']
    elif details['error_type'] in {'rate_limit_error', 'tokens', 'requests'}:
        reason = messages['rate_limit_exceeded']
    else:
        reason = 'API вернул HTTP 429 без уточнённой причины. Это может быть квота/баланс либо временный лимит запросов; проверьте Billing и Limits.'
    if code in {'rate_limit_exceeded', 'slow_down'} or (code is None and details['error_type'] in {'rate_limit_error', 'tokens', 'requests'}):
        if 'retry_after_seconds' in details:
            reason += f" Повторите не раньше чем через {details['retry_after_seconds']:g} с."
    return reason

def _generate(client, messages, question_id, allowed_ids, model=None):
    model = model or MODEL_NAME
    params = {'model':model, 'input':messages, 'max_output_tokens':SETTINGS['max_output_tokens'],
              'store':False, 'truncation':'disabled',
              'text':{'format':{'type':'json_schema','name':'model_output','strict':True,
                                'schema':output_schema(question_id,allowed_ids)}}}
    if model.startswith(('gpt-5', 'gpt-6')):
        params['reasoning'] = {'effort':'low'}
    started = time.perf_counter()
    try:
        response = client.responses.create(**params)
    except APIError as exc:
        # SDK exception strings may contain server echoes of credentials. Never
        # persist or display them; report only a safe category and status code.
        details = api_error_details(exc)
        if isinstance(exc, AuthenticationError):
            reason = 'API отклонил ключ. Проверьте OPENAI_API_KEY в .env.'
        elif isinstance(exc, RateLimitError):
            reason = rate_limit_message(details)
        elif isinstance(exc, APITimeoutError):
            reason = 'Истекло время ожидания ответа API.'
        elif isinstance(exc, APIConnectionError):
            reason = 'Не удалось подключиться к OpenAI API. Проверьте подключение и настройки прокси.'
        else:
            reason = 'Ошибка OpenAI API. Проверьте доступ к модели и параметры запроса.'
        status = getattr(exc,'status_code',None)
        suffix = (f' HTTP {status}.' if status else '')
        if details['error_code']:
            suffix += f" Код: {details['error_code']}."
        raise LLMError(reason + suffix,
                       usage={'response_seconds':time.perf_counter()-started}, metadata=details) from None
    usage = {'response_seconds':time.perf_counter()-started}
    if response.usage:
        usage.update(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                     total_tokens=response.usage.total_tokens)
        details = response.usage.input_tokens_details
        if details and details.cached_tokens is not None:
            usage['cached_input_tokens'] = details.cached_tokens
        details = response.usage.output_tokens_details
        if details and details.reasoning_tokens is not None:
            usage['reasoning_tokens'] = details.reasoning_tokens
    metadata = {'response_id':response.id, 'model':response.model, 'status':response.status,
                'request_id':getattr(response,'_request_id',None)}
    raw = response.output_text
    refusals = [content.refusal for item in response.output for content in (getattr(item,'content',None) or [])
                if getattr(content,'type',None)=='refusal']
    if refusals:
        raise LLMError('Модель отказалась отвечать на запрос.', raw_response='\n'.join(refusals), usage=usage, metadata=metadata)
    if response.status != 'completed':
        raise LLMError('API вернул незавершённый ответ. Проверьте лимит выходных токенов.',
                       raw_response=raw or None, usage=usage, metadata=metadata)
    if not raw:
        raise LLMError('API не вернул текст ответа.', usage=usage, metadata=metadata)
    return raw, usage, metadata

def total_usage(attempts):
    keys = ('input_tokens','output_tokens','total_tokens','cached_input_tokens','reasoning_tokens','response_seconds')
    return {**{key:sum(a['usage'].get(key,0) for a in attempts) for key in keys
               if any(key in a['usage'] for a in attempts)},
            'attempt_count':len(attempts), 'retry_count':max(0,len(attempts)-1)}

def infer(question, context, allowed_ids, provider='api_key', model=None, auth_profile=None):
    if not allowed_ids:
        raise LLMError('Контекст не содержит доступных утверждений.')
    if provider == 'chatgpt':
        from src.chatgpt_inference import infer_chatgpt
        return infer_chatgpt(question, context, allowed_ids, model or MODEL_NAME, auth_profile)
    if provider != 'api_key':
        raise LLMError('Неизвестный способ авторизации.')
    client = create_client()
    messages = make_messages(question, context, allowed_ids)
    attempts = []
    try:
        for index in range(SETTINGS['schema_retries']+1):
            try:
                raw, usage, metadata = _generate(client,messages,question.question_id,allowed_ids,model=model)
            except LLMError as exc:
                attempts.append({'attempt':index+1,'raw_response':exc.raw_response,'error':str(exc),
                                 'usage':exc.usage,'metadata':exc.metadata})
                exc.attempts, exc.usage = attempts,total_usage(attempts)
                raise
            normalized, changes = normalize_response(raw)
            attempt = {'attempt':index+1,'raw_response':raw,'usage':usage,'error':None,
                       'normalizations':changes,'normalized_response':normalized if changes else None,
                       'metadata':metadata}
            attempts.append(attempt)
            try:
                output = parse_output(raw,question.question_id,allowed_ids)
            except LLMError as exc:
                attempt['error'] = str(exc)
                if index == SETTINGS['schema_retries']:
                    exc.attempts,exc.usage,exc.metadata = attempts,total_usage(attempts),metadata
                    raise
                messages += [{'role':'assistant','content':raw},
                             {'role':'user','content':'Исправь полный ответ по исходному досье и JSON-схеме. '
                              +str(exc)+' Не добавляй фактов. Допустимые claim_id: '+', '.join(allowed_ids)}]
                continue
            return Inference(output,raw,total_usage(attempts),metadata['model'],attempts,metadata)
    finally:
        client.close()

def run_llm_inference(question, context, allowed_ids, provider=None, model=None):
    model, settings = configuration(provider, model)
    return infer(question,context,allowed_ids,provider='chatgpt' if settings['provider']=='chatgpt' else 'api_key',
                 model=model,auth_profile=settings.get('auth_profile')).output
