"""Public-client OAuth for local ChatGPT plan usage. No Codex/browser credential scraping."""
import base64
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

from filelock import FileLock, Timeout
import httpx
import jwt

ROOT = Path(__file__).resolve().parents[1]
ISSUER = 'https://auth.openai.com'
AUTHORIZE = ISSUER + '/api/accounts/authorize'
TOKEN = ISSUER + '/api/accounts/oauth/token'
JWKS = ISSUER + '/.well-known/jwks.json'
DISCOVERY = ISSUER + '/.well-known/openid-configuration'
RESOURCE = 'https://api.openai.com/v1'
DIRECT_SCOPE = 'chatgpt.tokens.use.direct'
SCOPES = 'openid profile email offline_access resource.invoke ' + DIRECT_SCOPE
USAGE_URL = 'https://chatgpt.com/settings/usage'
TERMINAL_REFRESH = {'invalid_grant', 'invalid_refresh_token', 'token_expired', 'refresh_token_expired',
                    'refresh_token_invalidated', 'refresh_token_reused'}


class AuthError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code


def _dpapi(data, decrypt=False):
    if os.name != 'nt':
        raise AuthError('Защищённое хранилище этой версии использует Windows DPAPI. Запустите приложение в Windows.')
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if decrypt:
        operation = crypt.CryptUnprotectData
        operation.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        ok = operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result))
    else:
        operation = crypt.CryptProtectData
        operation.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
        ok = operation(ctypes.byref(source), 'AI Detective ChatGPT session', None, None, None, 1, ctypes.byref(result))
    if not ok:
        raise AuthError('Не удалось открыть защищённое хранилище ChatGPT текущего пользователя Windows.')
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        kernel.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


class AuthStore:
    def __init__(self, directory=ROOT / '.chatgpt-auth'):
        self.directory = Path(directory)
        self.path = self.directory / 'session.bin'

    @contextmanager
    def lock(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with FileLock(str(self.directory / 'session.lock'), timeout=60):
                yield
        except Timeout:
            raise AuthError('Сессия ChatGPT занята другим процессом. Повторите действие позже.') from None
        except OSError as exc:
            raise AuthError(f'Не удалось открыть защищённое хранилище ChatGPT (код ОС {exc.errno}).') from exc

    def read(self):
        if not self.path.exists():
            return {'version': 1, 'host_id': None, 'active': None, 'profiles': {}}
        try:
            state = json.loads(_dpapi(self.path.read_bytes(), decrypt=True))
            if state.get('version') != 1 or not isinstance(state.get('profiles'), dict):
                raise ValueError('Unknown storage format')
            return state
        except (OSError, ValueError) as exc:
            raise AuthError('Не удалось прочитать хранилище ChatGPT. Токены не изменены.') from None

    def write(self, state):
        data = _dpapi(json.dumps(state, ensure_ascii=False).encode('utf-8'))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.bin', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _public(profile):
    fingerprint = hashlib.sha256((profile['issuer'] + '|' + profile['subject'] + '|' + profile['client_id']).encode()).hexdigest() if profile.get('subject') else None
    return {'profile_id': profile['client_id'], 'email': profile.get('email', ''),
            'auth_profile': fingerprint,
            'label': (profile.get('email') or 'Аккаунт ChatGPT') + ' · ' + profile['client_id'][-8:],
            'signed_in': bool(profile.get('access_token')),
            'sharing': bool(profile.get('access_token') and DIRECT_SCOPE in profile.get('scopes', []))}


class ChatGPTAuth:
    def __init__(self, store=None, client=None):
        self.store = store or AuthStore()
        self.client = client

    def profiles(self):
        with self.store.lock():
            state = self.store.read()
            return {'active': state['active'], 'profiles': [_public(p) for p in state['profiles'].values()]}

    def select(self, profile_id):
        with self.store.lock():
            state = self.store.read()
            if profile_id not in state['profiles']:
                raise AuthError('Аккаунт ChatGPT не найден.')
            state['active'] = profile_id
            self.store.write(state)

    def host_id(self):
        with self.store.lock():
            state = self.store.read()
            if not state['host_id']:
                state['host_id'] = 'urn:uuid:' + str(uuid.uuid4())
                self.store.write(state)
            return state['host_id']

    def _request(self, method, url, **kwargs):
        try:
            if self.client is not None:
                response = self.client.request(method, url, **kwargs)
            else:
                with httpx.Client(timeout=20, follow_redirects=False) as client:
                    response = client.request(method, url, **kwargs)
        except httpx.HTTPError:
            raise AuthError('Не удалось связаться с сервисом входа ChatGPT. Проверьте сеть и повторите позже.') from None
        if response.status_code >= 300:
            code = None
            try:
                body = response.json()
                error = body.get('error') if isinstance(body, dict) else None
                candidate = error.get('code') if isinstance(error, dict) else error
                if candidate in TERMINAL_REFRESH | {'invalid_client', 'access_denied'}:
                    code = candidate
            except (ValueError, TypeError):
                pass
            raise AuthError(f'Сервис входа ChatGPT вернул HTTP {response.status_code}. Повторите вход или проверьте доступность сервиса.', code)
        return response

    def _json(self, method, url, **kwargs):
        try:
            value = self._request(method, url, **kwargs).json()
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except ValueError:
            raise AuthError('Сервис входа вернул некорректный ответ.') from None

    def validate_identity(self, token, client_id, nonce=None, expected=None):
        try:
            header = jwt.get_unverified_header(token)
            if header.get('alg') != 'RS256':
                raise ValueError()
            keys = self._json('GET', JWKS).get('keys', [])
            if not isinstance(keys, list):
                raise ValueError()
            matching = [key for key in keys if isinstance(key, dict) and key.get('kid') == header.get('kid') and key.get('kty') == 'RSA']
            if not header.get('kid') or len(matching) != 1:
                raise ValueError()
            signing_key = jwt.PyJWK.from_dict(matching[0]).key
            required = ['iss', 'sub', 'aud', 'exp', 'iat'] + (['nonce'] if nonce is not None else [])
            claims = jwt.decode(token, signing_key, algorithms=['RS256'], issuer=ISSUER, audience=client_id,
                                leeway=30, options={'require': required})
            if not isinstance(claims.get('sub'), str) or not claims['sub']:
                raise ValueError()
            if claims.get('email') is not None and not isinstance(claims['email'], str):
                raise ValueError()
            if nonce is not None and (not isinstance(claims.get('nonce'), str) or not hmac.compare_digest(claims['nonce'], nonce)):
                raise ValueError()
            if 'azp' in claims and claims['azp'] != client_id:
                raise ValueError()
            if isinstance(claims['aud'], list) and len(claims['aud']) > 1 and claims.get('azp') != client_id:
                raise ValueError()
            if expected and (claims['iss'], claims['sub']) != (expected['issuer'], expected['subject']):
                raise ValueError()
            return claims
        except (jwt.PyJWTError, ValueError, TypeError, KeyError):
            raise AuthError('Не удалось проверить личность аккаунта ChatGPT. Активная сессия не заменена.') from None

    def _credentials(self, response, previous=None):
        previous = previous or {}
        scopes = response.get('scope')
        if 'scope' in response and not isinstance(scopes, str):
            raise AuthError('Сервис входа вернул некорректный список разрешений.')
        scopes = scopes.split() if isinstance(scopes, str) else previous.get('scopes', [])
        access = response.get('access_token')
        refresh = response.get('refresh_token')
        lifetime = response.get('expires_in', 0)
        token_type = response.get('token_type')
        if not isinstance(token_type, str) or token_type.lower() != 'bearer':
            raise AuthError('Сервис входа вернул неподдерживаемый тип токена.')
        if access is not None and (not isinstance(access, str) or not access):
            raise AuthError('Сервис входа не вернул действительные учётные данные.')
        if DIRECT_SCOPE in scopes and not access:
            raise AuthError('Разрешение на тариф получено без токена доступа. Повторите вход.')
        if isinstance(lifetime, bool) or not isinstance(lifetime, (int, float)) or not math.isfinite(lifetime) or lifetime < 0 or (access and lifetime == 0):
            raise AuthError('Некорректный срок действия сессии ChatGPT.')
        if refresh is not None and (not isinstance(refresh, str) or not refresh):
            raise AuthError('Некорректный ответ обновления сессии.')
        return {'access_token': access, 'refresh_token': refresh or previous.get('refresh_token'),
                'id_token': response.get('id_token') or previous.get('id_token'),
                'token_type': 'Bearer', 'scopes': scopes, 'expires_at': time.time() + lifetime,
                'earliest_refresh_at': response.get('earliest_refresh_at', 0)}

    def begin(self, profile_id=None, consent=False):
        with self.store.lock():
            state = self.store.read()
            profile = state['profiles'].get(profile_id) if profile_id else None
            if profile_id and profile is None:
                raise AuthError('Аккаунт ChatGPT не найден.')
            if profile:
                profile = {key: profile[key] for key in ('client_id', 'issuer', 'subject') if key in profile}
        return LoginFlow(self, profile, consent)

    def complete(self, flow, callback):
        if not isinstance(callback.get('state'), str) or not hmac.compare_digest(callback['state'], flow.state):
            raise AuthError('Неверный state при завершении входа.')
        returned_id = callback.get('client_id')
        if flow.profile and returned_id and returned_id != flow.profile['client_id']:
            raise AuthError('Callback вернул другой client_id. Аккаунт не изменён.')
        client_id = flow.profile['client_id'] if flow.profile else returned_id
        if not client_id or client_id == 'dynamic_agent_client' or not re_valid_id(client_id):
            raise AuthError('Регистрация не завершена: сервер не выдал client_id.')
        # Retain a newly issued ID even if its one-time code expires. This record
        # is not active and cannot authorize inference until identity is verified.
        with self.store.lock():
            state = self.store.read()
            state['profiles'].setdefault(client_id, {'client_id': client_id})
            self.store.write(state)
        response = self._json('POST', TOKEN, data={'grant_type': 'authorization_code', 'client_id': client_id,
                    'code': callback['code'], 'code_verifier': flow.verifier, 'redirect_uri': flow.redirect_uri,
                    'resource': RESOURCE})
        identity = self.validate_identity(response.get('id_token'), client_id, flow.nonce,
                                          flow.profile if flow.profile and flow.profile.get('subject') else None)
        credentials = self._credentials(response)
        with self.store.lock():
            state = self.store.read()
            existing = state['profiles'].get(client_id, {})
            if existing.get('subject') and (existing['issuer'], existing['subject']) != (identity['iss'], identity['sub']):
                raise AuthError('Нельзя заменить регистрацию другим аккаунтом.')
            profile = {'client_id': client_id, 'issuer': identity['iss'], 'subject': identity['sub'],
                       'email': identity.get('email') or '', **credentials}
            state['profiles'][client_id] = profile
            state['active'] = client_id
            self.store.write(state)
        return _public(profile)

    def access(self):
        # The file lock covers refresh + atomic replacement, including across
        # Streamlit and CLI processes sharing the rotating refresh token.
        with self.store.lock():
            state = self.store.read()
            profile = state['profiles'].get(state['active'], {})
            if not profile.get('subject') or not profile.get('access_token'):
                raise AuthError('Войдите через Continue with ChatGPT.')
            if DIRECT_SCOPE not in profile.get('scopes', []):
                raise AuthError('Вход выполнен без разрешения Use your ChatGPT plan. Разрешите использование тарифа.')
            now = time.time()
            if now >= profile.get('expires_at', 0) - 60:
                earliest = profile.get('earliest_refresh_at', 0)
                if isinstance(earliest, (int, float)) and earliest > now:
                    if now >= profile['expires_at']:
                        raise AuthError('Сессия истекла; обновление пока недоступно. Повторите вход позже.')
                else:
                    if not profile.get('refresh_token'):
                        raise AuthError('Сессия истекла. Повторите вход через ChatGPT.')
                    try:
                        response = self._json('POST', TOKEN, data={'grant_type': 'refresh_token',
                            'client_id': profile['client_id'], 'refresh_token': profile['refresh_token'], 'resource': RESOURCE})
                    except AuthError as exc:
                        if exc.code in TERMINAL_REFRESH:
                            self._clear(profile)
                            self.store.write(state)
                            raise AuthError('Сессия ChatGPT больше не действует. Войдите снова.') from None
                        raise
                    if response.get('id_token'):
                        self.validate_identity(response['id_token'], profile['client_id'], expected=profile)
                    profile.update(self._credentials(response, profile))
                    self.store.write(state)
                    if DIRECT_SCOPE not in profile['scopes'] or not profile.get('access_token'):
                        raise AuthError('Разрешение на использование тарифа отсутствует. Повторите вход.')
            fingerprint = hashlib.sha256((profile['issuer'] + '|' + profile['subject'] + '|' + profile['client_id']).encode()).hexdigest()
            return profile['access_token'], fingerprint

    @staticmethod
    def _clear(profile):
        for key in ('access_token', 'refresh_token', 'id_token', 'expires_at', 'earliest_refresh_at'):
            profile.pop(key, None)
        profile['scopes'] = []

    def logout(self, profile_id=None):
        with self.store.lock():
            state = self.store.read()
            profile = state['profiles'].get(profile_id or state['active'])
            if not profile:
                return True
            confirmed = not bool(profile.get('refresh_token'))
            if profile.get('refresh_token'):
                for attempt in range(2):
                    try:
                        config = self._json('GET', DISCOVERY)
                        endpoint = config.get('revocation_endpoint', '')
                        parsed = urlsplit(endpoint)
                        if config.get('issuer') != ISSUER or parsed.scheme != 'https' or parsed.netloc != 'auth.openai.com':
                            raise AuthError('Некорректный адрес завершения сессии.')
                        self._request('POST', endpoint, data={'token': profile['refresh_token'],
                            'token_type_hint': 'refresh_token', 'client_id': profile['client_id']})
                        confirmed = True
                        break
                    except AuthError:
                        if attempt == 0:
                            time.sleep(.25)
            self._clear(profile)
            self.store.write(state)
            return confirmed


def re_valid_id(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and all(c.isascii() and (c.isalnum() or c in '_-') for c in value)


class LoginFlow:
    """One ephemeral loopback callback. No OAuth codes or query strings are logged."""
    def __init__(self, service, profile=None, consent=False, lifetime=600):
        self.service, self.profile = service, profile
        self.state, self.nonce, self.verifier = (secrets.token_urlsafe(32) for _ in range(3))
        self.deadline = time.monotonic() + lifetime
        self.received = threading.Event()
        self.stopped = threading.Event()
        self.mutex = threading.Lock()
        self.callback = None
        self.used = False
        host_id = service.host_id()
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                if parsed.path != '/auth/callback' or len(self.path) > 16384:
                    self.send_error(404)
                    return
                query = parse_qs(parsed.query, keep_blank_values=True)
                valid = (all(len(values) == 1 for values in query.values()) and
                         isinstance(query.get('state', [None])[0], str) and
                         hmac.compare_digest(query['state'][0].encode('utf-8'), owner.state.encode('ascii')))
                if not valid or time.monotonic() > owner.deadline:
                    self.send_error(400, 'Invalid authorization state')
                    return
                with owner.mutex:
                    if owner.callback is not None:
                        self.send_error(409)
                        return
                    if not query.get('code', [''])[0] and not query.get('error', [''])[0]:
                        self.send_error(400, 'Missing authorization result')
                        return
                    owner.callback = {key: values[0] for key, values in query.items() if key in {'code', 'state', 'client_id', 'error'}}
                body = 'Ответ входа получен. Вернитесь в приложение AI Detective или терминал, чтобы завершить вход.'.encode('utf-8')
                self.send_response(200)
                for key, value in [('Content-Type', 'text/plain; charset=utf-8'), ('Cache-Control', 'no-store'),
                                   ('Referrer-Policy', 'no-referrer'), ('X-Content-Type-Options', 'nosniff')]:
                    self.send_header(key, value)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                owner.received.set()
                owner.stopped.set()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.timeout = .25
        self.redirect_uri = f'http://127.0.0.1:{self.server.server_port}/auth/callback'
        challenge = base64.urlsafe_b64encode(hashlib.sha256(self.verifier.encode()).digest()).rstrip(b'=').decode()
        params = {'client_id': profile['client_id'] if profile else 'dynamic_agent_client',
                  'ext_agent_host_id': host_id, 'response_type': 'code', 'redirect_uri': self.redirect_uri,
                  'scope': SCOPES, 'resource': RESOURCE, 'state': self.state, 'nonce': self.nonce,
                  'code_challenge_method': 'S256', 'code_challenge': challenge}
        if not profile:
            params['agent_name_hint'] = 'AI Detective'
        if consent:
            params['prompt'] = 'consent'
        self.url = AUTHORIZE + '?' + urlencode(params)
        self.thread = threading.Thread(target=self._serve, daemon=True, name='chatgpt-loopback')
        self.thread.start()

    def _serve(self):
        try:
            while not self.stopped.is_set() and time.monotonic() < self.deadline:
                self.server.handle_request()
        finally:
            self.server.server_close()

    def finish(self):
        with self.mutex:
            if self.used:
                raise AuthError('Эта попытка входа уже завершена. Начните новую.')
            if not self.received.is_set():
                raise AuthError('Ожидается вход в браузере.' if time.monotonic() < self.deadline else 'Время входа истекло. Начните снова.')
            self.used = True
            callback = self.callback
        try:
            if callback.get('error'):
                raise AuthError('Вход или разрешение использования тарифа отменены пользователем.')
            return self.service.complete(self, callback)
        finally:
            self.close()
            self.callback = None
            self.verifier = ''

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=1)
        with self.mutex:
            self.used = True
            self.callback = None
            self.verifier = self.state = self.nonce = ''
