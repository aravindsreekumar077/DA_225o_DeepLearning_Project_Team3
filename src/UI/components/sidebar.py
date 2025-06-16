import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ SLAM Controls")
        uploaded_file = st.file_uploader("Attach a file (optional):", type=["txt", "pdf", "docx"])
        file_content = ""
        file_name = ""

        # ✅ Scratchpad Toggle Button
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            if st.button("📝", key="scratchpad_button"):
                st.session_state.scratchpad_enabled = not st.session_state.get("scratchpad_enabled", False)
                st.session_state.scratchpad_toggled_once = False
                st.rerun()
        with col2:
            enabled = st.session_state.get("scratchpad_enabled", False)
            badge_text = "Scratchpad Enabled" if enabled else "Scratchpad Disabled"
            badge_color = "#28a745" if enabled else "#888888"
            st.markdown(
                f"<span style='color:{badge_color}; font-weight:600;'>{badge_text}</span>",
                unsafe_allow_html=True
            )

        if uploaded_file is not None:
            file_name = uploaded_file.name
            try:
                file_content = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        if st.button("🔄 New Chat"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

        return uploaded_file, file_content, file_name
