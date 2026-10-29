"""ChatGPT plan transport: public Responses endpoint, OAuth and terminal SSE events."""
import threading
import time
import httpx
from openai import OpenAI, APIError
from src.chatgpt_auth import ChatGPTAuth, AuthError, RESOURCE
from src.llm import (Inference, LLMError, SETTINGS, make_messages, output_schema,
                     parse_output, normalize_response, total_usage, api_error_details)

_catalog = {}
_catalog_lock = threading.Lock()
PLAN_ERRORS = {
    'subscription_sharing_usage_limit_exceeded': 'Достигнут лимит подписки ChatGPT. Откройте ChatGPT → Settings → Usage.',
    'subscription_sharing_usage_unavailable': 'Использование тарифа ChatGPT сейчас недоступно. Проверьте Settings → Usage.',
    'subscription_sharing_unsupported_capability': 'Выбранная возможность недоступна через подписку. Обновите список моделей и параметры запроса.',
    'subscription_sharing_route_not_supported': 'Маршрут запроса не поддерживается для подписки ChatGPT.',
    'subscription_sharing_invalid_user': 'Не удалось подтвердить аккаунт ChatGPT. Проверьте выбранный аккаунт и повторите вход при необходимости.',
    'chatpass_v2_scope_not_authorized': 'Аккаунт не разрешил использование тарифа. Нажмите «Разрешить использование тарифа».',
    'chatpass_v2_invalid_authorization_context': 'Разрешение аккаунта не подходит для запроса. Повторите вход через ChatGPT.',
    'subscription_sharing_user_unavailable': 'Сервис аккаунтов ChatGPT временно недоступен. Повторите позже.',
}


def list_models(auth=None, refresh=False, expected_profile=None):
    auth = auth or ChatGPTAuth()
    token, profile = auth.access()
    if expected_profile and profile != expected_profile:
        raise AuthError('Аккаунт ChatGPT изменился. Повторите запуск с выбранным аккаунтом.')
    with _catalog_lock:
        cached = _catalog.get(profile)
        if not refresh and cached and time.monotonic()-cached[0] < 300:
            return list(cached[1])
    response = auth._json('GET', RESOURCE + '/models', headers={'Authorization': 'Bearer ' + token})
    models = [{'slug': item['slug'], 'display_name': item.get('display_name') or item['slug']}
              for item in response.get('models', []) if isinstance(item, dict)
              and item.get('visibility') == 'list' and isinstance(item.get('slug'), str)]
    if not models:
        raise AuthError('Этот аккаунт не вернул доступных моделей. Проверьте подписку и разрешение на её использование.')
    with _catalog_lock:
        _catalog[profile] = (time.monotonic(), models)
    return models


def _usage(response, started):
    usage = {'response_seconds': time.perf_counter()-started}
    values = getattr(response, 'usage', None)
    if values:
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            value = getattr(values, key, None)
            if value is not None:
                usage[key] = value
        for name, key in [('input_tokens_details', 'cached_tokens'), ('output_tokens_details', 'reasoning_tokens')]:
            value = getattr(getattr(values, name, None), key, None)
            if value is not None:
                usage['cached_input_tokens' if key == 'cached_tokens' else key] = value
    return usage


def _stream(client, messages, question_id, allowed_ids, model, profile):
    params = {'model': model, 'instructions': '\n'.join(m['content'] for m in messages if m['role'] == 'system'),
              'input': [m for m in messages if m['role'] != 'system'], 'store': False, 'stream': True,
              'text': {'format': {'type': 'json_schema', 'name': 'model_output', 'strict': True,
                                  'schema': output_schema(question_id, allowed_ids)}}}
    if model.startswith(('gpt-5', 'gpt-6')):
        params['reasoning'] = {'effort': 'low'}
    started = time.perf_counter()
    chunks, refusals = [], []
    metadata = {'provider': 'chatgpt', 'auth_profile': profile, 'model': model}
    try:
        with client.responses.create(**params) as stream:
            request_id = stream.response.headers.get('x-request-id')
            if request_id:
                metadata['request_id'] = request_id
            for event in stream:
                if event.type == 'response.output_text.delta':
                    chunks.append(event.delta)
                elif event.type == 'response.refusal.delta':
                    refusals.append(event.delta)
                elif event.type in {'response.completed', 'response.failed', 'response.incomplete'}:
                    response = event.response
                    usage = _usage(response, started)
                    metadata.update(response_id=response.id, model=response.model, status=response.status)
                    error = getattr(response, 'error', None)
                    code = getattr(error, 'code', None)
                    if code in PLAN_ERRORS:
                        metadata['error_code'] = code
                    if event.type != 'response.completed' or response.status != 'completed':
                        raise LLMError(PLAN_ERRORS.get(code, 'Запрос по подписке не завершён. Проверьте соединение и доступность модели.'),
                                       raw_response=''.join(chunks) or None, usage=usage, metadata=metadata)
                    final_refusals = [c.refusal for item in response.output for c in (getattr(item, 'content', None) or [])
                                      if getattr(c, 'type', None) == 'refusal']
                    if refusals or final_refusals:
                        raise LLMError('Модель отказалась отвечать.', raw_response=''.join(refusals) or '\n'.join(final_refusals),
                                       usage=usage, metadata=metadata)
                    raw = response.output_text or ''.join(chunks)
                    if not raw:
                        raise LLMError('Модель не вернула текст ответа.', usage=usage, metadata=metadata)
                    return raw, usage, metadata
                elif event.type == 'error':
                    code = getattr(event, 'code', None)
                    if code in PLAN_ERRORS:
                        metadata['error_code'] = code
                    raise LLMError(PLAN_ERRORS.get(code, 'Ошибка в потоке ответа ChatGPT.'), raw_response=''.join(chunks) or None,
                                   usage=_usage(None, started), metadata=metadata)
    except APIError as exc:
        details = api_error_details(exc)
        metadata.update(details)
        code = details.get('error_code')
        reason = PLAN_ERRORS.get(code, 'Запрос через подписку ChatGPT отклонён или сервис недоступен. Проверьте аккаунт и Settings → Usage.')
        status = details.get('http_status')
        raise LLMError(reason + (f' HTTP {status}.' if status else ''), raw_response=''.join(chunks) or None,
                       usage=_usage(None, started), metadata=metadata) from None
    except httpx.HTTPError:
        raise LLMError('Соединение с ChatGPT прервано. Повторите запрос вручную.', raw_response=''.join(chunks) or None,
                       usage=_usage(None, started), metadata=metadata) from None
    raise LLMError('Поток завершился без response.completed. Ответ не считается успешным.',
                   raw_response=''.join(chunks) or None, usage=_usage(None, started), metadata=metadata)


def infer_chatgpt(question, context, allowed_ids, model, expected_profile=None, auth=None):
    auth = auth or ChatGPTAuth()
    try:
        token, profile = auth.access()
        if expected_profile and profile != expected_profile:
            raise AuthError('Аккаунт ChatGPT изменился. Повторите запуск с выбранным аккаунтом.')
        models = list_models(auth, expected_profile=profile)
        if model not in {m['slug'] for m in models}:
            raise AuthError('Модель недоступна выбранному аккаунту ChatGPT. Обновите список и выберите доступную модель.')
    except AuthError as exc:
        raise LLMError(str(exc)) from None
    client = OpenAI(api_key=token, base_url=RESOURCE, timeout=SETTINGS['timeout_seconds'], max_retries=0)
    messages, attempts = make_messages(question, context, allowed_ids), []
    try:
        for index in range(SETTINGS['schema_retries']+1):
            try:
                raw, usage, metadata = _stream(client, messages, question.question_id, allowed_ids, model, profile)
            except LLMError as exc:
                attempts.append({'attempt': index+1, 'raw_response': exc.raw_response, 'usage': exc.usage,
                                 'metadata': exc.metadata, 'error': str(exc)})
                exc.attempts, exc.usage = attempts, total_usage(attempts)
                raise
            normalized, changes = normalize_response(raw)
            attempt = {'attempt': index+1, 'raw_response': raw, 'usage': usage, 'metadata': metadata,
                       'normalizations': changes, 'normalized_response': normalized if changes else None, 'error': None}
            attempts.append(attempt)
            try:
                output = parse_output(raw, question.question_id, allowed_ids)
            except LLMError as exc:
                attempt['error'] = str(exc)
                if index == SETTINGS['schema_retries']:
                    exc.attempts, exc.usage, exc.metadata = attempts, total_usage(attempts), metadata
                    raise
                messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                             'Исправь ответ по исходному досье и JSON-схеме. ' + str(exc)}]
                continue
            return Inference(output, raw, total_usage(attempts), metadata['model'], attempts, metadata)
    finally:
        client.close()
