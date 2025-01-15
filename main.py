import os
import sys

# set working directory to LyricChat repo root (to identify .env file)
script_path = os.path.dirname(os.path.abspath(__file__))
# add src folder to sys.path
src_folder = os.path.join(script_path, "src")
sys.path.append(src_folder)

import rag
from llm import human_msg, AI_msg
import pandas as pd
import streamlit as st
from streamlit_player import st_player  # embedd music/video

# import extra_streamlit_components as stx  # extra func
import uuid  # unique ID
from datetime import datetime
from io import BytesIO

# Set up environment variables
os.environ["Qdrant_API_KEY"] = st.secrets["Qdrant_API_KEY"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


def get_timestamp():
    current_timestamp = datetime.now()
    return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def video_msg(video_url):
    return {"role": "video", "content": video_url}


# chat history file:
chat_history_dir = "data/chat_history"


def main():  # streamlit run main.py
    setup_interface()  # setup webpage
    display_chat_history()  # display chat history
    chatbot = cache_LyricChat()  # initialize RAG-LLM
    # start the conversation:
    lbl_chat_input = {
        "en": "Share what's on your mind. Wonda will find the perfect song to match your mood!",
        "tw": "說說你的心情吧! 幻答會幫你找到最適合的歌",
    }
    user_input = st.chat_input(lbl_chat_input[st.session_state.selected_language])
    if user_input:  # if user type something
        with st.chat_message("Human"):
            st.markdown(user_input)
        # Stage one, do sentiment analysis for DB filtering
        progress_bar = st.progress(0, text="Identifying emotion...")
        chatbot.user_input = user_input

        chatbot.chain_classify(language=st.session_state.selected_language)

        if chatbot.temp_response is not None:  # LLM suggest don't continue workflow
            with st.chat_message("AI"):
                progress_bar.progress(
                    40,
                    text="Insufficient or unrelated input detected; pausing workflow.",
                )
                st.markdown(chatbot.temp_response)
                model_response = chatbot.temp_response
        else:  # Stage two, song recommendation
            progress_bar.progress(
                40, text=f"Classified emotions: {chatbot.classify_res.emotions}"
            )
            with st.chat_message("AI"):
                response = chatbot.chain_rag(
                    top_r=3, stream=True, language=st.session_state.selected_language
                )
                model_response = st.write_stream(rag.yield_stream(response))
                st_player(chatbot.youtube_link)
        # save chat history:
        st.session_state.chat_history.append(human_msg(user_input))  # save user input
        st.session_state.chat_history.append(AI_msg(model_response))
        if chatbot.youtube_link is not None:
            st.session_state.chat_history.append(video_msg(chatbot.youtube_link))
        # update LLM memory to latest 3 conversations
        chatbot.update_memory(chat_history=st.session_state.chat_history, memory=3)
        st.rerun()  # update download button for chat history


@st.cache_resource  # Use cache_resource to avoid reload embedding model
def cache_LyricChat():
    return rag.LyricRAG()


def setup_interface():
    # TODO: set page configs
    # st.title("LyricChat: Turn your Feelings into Melody")
    st.set_page_config(page_title="LyricChat", page_icon="🎵", layout="centered")
    with st.container():
        # Push language selector to the right
        left_col, right_col = st.columns([4.5, 1])  # 80% empty space, 20% for language
        with right_col:
            languages = {"English": "en", "繁體中文": "tw"}
            language_key = st.selectbox(
                label="Language",
                options=list(languages.keys()),
                index=0,
                label_visibility="collapsed",
            )
            selected_language = languages[language_key]
            st.session_state.selected_language = selected_language
        with left_col:
            lbl_title = {"en": "Turn feelings into melody", "tw": "化感受為旋律"}
            st.markdown(
                f"""
        <h2 style='text-align: center; font-size: 32px; color: #333333;'>LyricChat: {lbl_title[selected_language]} 🎼</h2>
        """,
                unsafe_allow_html=True,
            )

    # create buffer:
    if "session_ID" not in st.session_state:  # initialize a session w chat history
        st.session_state.chat_history = []
        st.session_state.session_ID = uuid.uuid4().hex
    # set colors:
    custom_css = """
    <style>
    .stApp {
        background-color: #E6F3FF;
    }
    .stButton>button {
        background-color: #4DA8DA;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #3890C8;  /* Slightly darker on hover */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:disabled {
        background-color: #B8D4E3;  /* Lighter color for disabled state */
        cursor: not-allowed;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
    }
    .stSelectbox>div>div>select {
        background-color: #FFFFFF;
        border: 1px solid #4DA8DA;
        border-radius: 4px;
        color: #2C3E50;  /* Darker text for better readability */
    }
    .stHeader {
        background-color: #4DA8DA;
        color: white;
    }
    .element-container blockquote {
        background-color: #EAF4F9;
        border-left: 5px solid #4DA8DA;
        padding: 10px;
        margin: 10px 0;
    }
    .chat-message {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    /* Style for link button to match other buttons */
    .stLinkButton>a {
        background-color: #4DA8DA !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stLinkButton>a:hover {
        background-color: #3890C8 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column widths to be equal
    lbl_history = {"en": "Chat History", "tw": "聊天紀錄"}
    with col1:
        df = pd.DataFrame(st.session_state.chat_history)
        if df.shape[0] > 0:
            output = BytesIO()
            # Write the DataFrame to the BytesIO object
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(
                    writer, index=False, sheet_name="Sheet1"
                )  # Save DataFrame to Excel

            st.download_button(
                label=lbl_history[selected_language],
                data=output.getvalue(),
                file_name=f"chat_history_{st.session_state.session_ID}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button(
                lbl_history[selected_language], disabled=True, use_container_width=True
            )
    with col2:
        lbl_feedback = {"en": "Give us your feedback!", "tw": "回饋表單"}
        st.link_button(
            lbl_feedback[selected_language],
            "https://forms.gle/Xq2vo4TcVa4UMyXNA",
            use_container_width=True,
        )

    with col3:
        lbl_restart = {"en": "Restart the Chat", "tw": "重新開始"}
        if st.button(lbl_restart[selected_language], use_container_width=True):
            restart_conversation()


def restart_conversation():
    """Helper for setup_config() to restart the conversation"""
    st.session_state.chat_history = []
    st.session_state.session_ID = uuid.uuid4().hex


def display_chat_history():
    # Load chat history into conversation
    if "chat_history" in st.session_state:  # initialize a session w chat history
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "assistant":
                with st.chat_message("AI"):
                    st.markdown(msg["content"])
            elif msg["role"] == "user":
                with st.chat_message("Human"):
                    st.markdown(msg["content"])
            elif msg["role"] == "video":
                with st.chat_message("AI"):
                    st_player(msg["content"], key=f"v{idx}")


if __name__ == "__main__":
    main()
