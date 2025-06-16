import streamlit as st

def render_chat_input(agent_api, uploaded_file, file_content, file_name):
    if "prev_scratchpad_state" not in st.session_state:
        st.session_state.prev_scratchpad_state = st.session_state.get("scratchpad_enabled", False)

    current_state = st.session_state.get("scratchpad_enabled", False)
    scratchpad_just_disabled = (
        st.session_state.prev_scratchpad_state and not current_state
    )
    st.session_state.prev_scratchpad_state = current_state

    if scratchpad_just_disabled:
        with st.chat_message("assistant"):
            st.markdown("🧠 Summarizing your notes...")
            result = agent_api.toggle_scratchpad(False)
            st.markdown(result["response"])
            st.session_state.chat_history.append({"role": "assistant", "content": result["response"]})

    if st.session_state.get("scratchpad_toggled_once", False) is False and current_state:
        agent_api.toggle_scratchpad(True)
        st.session_state.scratchpad_toggled_once = True

    if st.session_state.get("clear_input", False):
        st.session_state.input = ""
        st.session_state.clear_input = False

    user_input = st.chat_input("Ask the agent something...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if current_state:
            agent_api.get_agent_response(user_input, file_content, file_name)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = agent_api.get_agent_response(user_input, file_content, file_name)
                    response = result["response"]
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
