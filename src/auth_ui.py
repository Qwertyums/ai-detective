"""Local account controls. Streamlit only receives public profile information."""
import streamlit as st
import time
from src.llm import DEFAULT_PROVIDER, MODEL_NAME, api_configured
from src.chatgpt_auth import ChatGPTAuth, AuthError, USAGE_URL
from src.chatgpt_inference import list_models


def connection_controls():
    st.sidebar.header('Подключение модели')
    provider = st.sidebar.radio('Источник лимитов', ['chatgpt', 'api_key'],
        index=1 if DEFAULT_PROVIDER == 'api_key' else 0,
        format_func=lambda p: 'Подписка ChatGPT' if p == 'chatgpt' else 'OpenAI API — отдельный биллинг')
    if provider == 'api_key':
        model = st.sidebar.text_input('Модель API', value=MODEL_NAME).strip()
        if not api_configured():
            st.sidebar.warning('Для отдельного API-биллинга нужен OPENAI_API_KEY в .env.')
        return provider, model, bool(api_configured() and model)

    auth = ChatGPTAuth()
    flash = st.session_state.pop('chatgpt_auth_notice', None)
    if flash:
        getattr(st.sidebar, flash['kind'])(flash['text'])
    try:
        accounts = auth.profiles()
    except AuthError as exc:
        st.sidebar.error(str(exc))
        return provider, MODEL_NAME, False
    profiles = {p['profile_id']: p for p in accounts['profiles']}
    options = ['new'] + list(profiles)
    current = accounts['active'] if accounts['active'] in profiles else 'new'
    selected = st.sidebar.selectbox('Аккаунт ChatGPT', options, index=options.index(current),
        key='chatgpt_account_' + str(accounts['active']),
        format_func=lambda key: 'Добавить аккаунт' if key == 'new' else profiles[key]['label'])
    if selected != 'new' and selected != accounts['active']:
        pending = st.session_state.pop('chatgpt_login_flow', None)
        if pending:
            pending.close()
        auth.select(selected)
        st.rerun()
    profile = profiles.get(selected)
    connect_label = ('Разрешить использование тарифа' if profile and profile['signed_in'] and not profile['sharing']
                     else 'Войти заново' if profile and profile['sharing'] else 'Continue with ChatGPT')
    if st.sidebar.button(connect_label):
        pending = st.session_state.pop('chatgpt_login_flow', None)
        if pending:
            pending.close()
        try:
            st.session_state['chatgpt_login_flow'] = auth.begin(None if selected == 'new' else selected,
                consent=bool(profile and profile['signed_in'] and not profile['sharing']))
        except (AuthError, OSError) as exc:
            st.sidebar.error(str(exc) if isinstance(exc, AuthError) else 'Не удалось запустить локальный обработчик входа.')

    flow = st.session_state.get('chatgpt_login_flow')
    if flow:
        st.sidebar.link_button('Открыть вход в ChatGPT', flow.url)
        st.sidebar.caption('В браузере разрешите Use your ChatGPT plan, затем вернитесь и проверьте вход. Ссылка действует 10 минут.')
        if st.sidebar.button('Проверить вход'):
            if not flow.received.is_set():
                st.sidebar.info('Ожидается завершение входа в браузере.' if time.monotonic() < flow.deadline
                                else 'Ссылка истекла. Отмените вход и начните его снова.')
            else:
                try:
                    with st.spinner('Проверка аккаунта ChatGPT...'):
                        result = flow.finish()
                    notice = {'kind': 'success' if result['sharing'] else 'warning',
                              'text': 'Подключено: ' + result['label'] + ('. Используется тариф ChatGPT.' if result['sharing'] else '. Разрешение на использование тарифа не предоставлено.')}
                    if result['sharing']:
                        try:
                            st.session_state['chatgpt_models_' + result['auth_profile']] = list_models(auth, refresh=True)
                        except AuthError:
                            notice = {'kind': 'warning', 'text': 'Вход выполнен. Список моделей пока не получен — нажмите «Обновить список моделей».'}
                    st.session_state['chatgpt_auth_notice'] = notice
                except AuthError as exc:
                    st.session_state['chatgpt_auth_notice'] = {'kind': 'error', 'text': str(exc)}
                finally:
                    flow.close()
                    st.session_state.pop('chatgpt_login_flow', None)
                st.rerun()
        if st.sidebar.button('Отменить вход'):
            flow.close()
            st.session_state.pop('chatgpt_login_flow', None)
            st.rerun()

    if profile and profile['signed_in']:
        st.sidebar.caption('Активный аккаунт: ' + profile['label'])
        if st.sidebar.button('Выйти из ChatGPT'):
            with st.spinner('Завершение сессии...'):
                confirmed = auth.logout(selected)
            st.session_state['chatgpt_auth_notice'] = {'kind': 'success' if confirmed else 'warning',
                'text': 'Выход выполнен.' if confirmed else 'Локальный выход выполнен. Отзыв на сервере не подтверждён; отключите приложение в настройках ChatGPT.'}
            st.rerun()
    st.sidebar.link_button('Управлять лимитами ChatGPT', USAGE_URL)
    if not profile or not profile['sharing']:
        st.sidebar.info('Войдите через ChatGPT и разрешите использование тарифа. API-ключ для этого режима не нужен.')
        return provider, MODEL_NAME, False
    st.sidebar.caption('Используется тариф ChatGPT. Расход учитывается в общих лимитах подписки.')
    catalog_key = 'chatgpt_models_' + profile['auth_profile']
    if st.sidebar.button('Обновить список моделей'):
        try:
            with st.spinner('Получение доступных моделей...'):
                st.session_state[catalog_key] = list_models(auth, refresh=True)
        except AuthError as exc:
            st.sidebar.error(str(exc))
            st.session_state.pop(catalog_key, None)
    catalog = st.session_state.get(catalog_key, [])
    if not catalog:
        st.sidebar.info('Загрузите список моделей кнопкой выше.')
        return provider, MODEL_NAME, False
    slugs = [m['slug'] for m in catalog]
    names = {m['slug']: m['display_name'] for m in catalog}
    model = st.sidebar.selectbox('Модель подписки', slugs, index=slugs.index(MODEL_NAME) if MODEL_NAME in slugs else 0,
                                format_func=lambda slug: names[slug], key='chatgpt_model_' + profile['auth_profile'])
    return provider, model, True
