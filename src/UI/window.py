import streamlit as st
from backend.backend_interface import BackendInterface
from components.sidebar import render_sidebar
from components.title import render_title
from components.chat_display import render_chat_history
from components.chat_input import render_chat_input

class SLAMAppWindow:
    def __init__(self):
        # Backend API setup
        if "agent_api" not in st.session_state:
            st.session_state.agent_api = BackendInterface()
        self.agent_api = st.session_state.agent_api
        self._init_session_state()

    def _init_session_state(self):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "clear_input" not in st.session_state:
            st.session_state.clear_input = False

    def run(self):
        st.set_page_config(
            page_title="SLAM - Smart Language Agentic Machine",
            page_icon="🤖",
            layout="wide"
        )

        # Sidebar
        uploaded_file, file_content, file_name = render_sidebar()

        # Title
        render_title()

        # Display previous chat
        render_chat_history()

        # Input section
        render_chat_input(
            self.agent_api,
            uploaded_file,
            file_content,
            file_name
        )
