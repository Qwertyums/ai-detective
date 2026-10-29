from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class PublicAuth:
    def __init__(self, sharing):
        self.sharing=sharing
    def profiles(self):
        return {'active':'oaiapp_test','profiles':[{'profile_id':'oaiapp_test','label':'Test account',
                 'auth_profile':'test-profile','signed_in':True,'sharing':self.sharing}]}


SCRIPT='''
import streamlit as st
from src.auth_ui import connection_controls
st.session_state['connection_result']=connection_controls()
'''


def test_subscription_is_ready_without_api_key_when_consent_exists():
    with patch('src.auth_ui.ChatGPTAuth',return_value=PublicAuth(True)), patch('src.auth_ui.api_configured',return_value=False), patch('src.auth_ui.DEFAULT_PROVIDER','chatgpt'):
        app=AppTest.from_string(SCRIPT)
        app.session_state['chatgpt_models_test-profile']=[{'slug':'gpt-6-luna','display_name':'Luna'}]
        app.run()
    assert not app.exception
    assert app.session_state['connection_result']==('chatgpt','gpt-6-luna',True)


def test_identity_without_plan_consent_is_not_ready():
    with patch('src.auth_ui.ChatGPTAuth',return_value=PublicAuth(False)), patch('src.auth_ui.DEFAULT_PROVIDER','chatgpt'):
        app=AppTest.from_string(SCRIPT).run()
    assert not app.exception
    assert app.session_state['connection_result'][2] is False
    assert any(button.label=='Разрешить использование тарифа' for button in app.button)


def test_api_key_mode_requires_explicit_selection_and_key():
    with patch('src.auth_ui.ChatGPTAuth',return_value=PublicAuth(True)), patch('src.auth_ui.api_configured',return_value=False), patch('src.auth_ui.DEFAULT_PROVIDER','chatgpt'):
        app=AppTest.from_string(SCRIPT).run()
        # Streamlit 1.32's test driver serializes formatted widgets by label.
        app.sidebar.selectbox[0].set_value('Test account')
        app.sidebar.radio[0].set_value(app.sidebar.radio[0].options[1]).run()
    assert not app.exception
    provider,_,ready=app.session_state['connection_result']
    assert provider=='api_key' and ready is False
