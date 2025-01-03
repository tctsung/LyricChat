import os
import sys

# set working directory to LyricChat repo root
print(f"Current working directory: {os.getcwd()}")

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import src.rag.rag as rag
from src.llm import human_msg, AI_msg
import pandas as pd
import streamlit as st
from streamlit_player import st_player  # embedd music/video
import re
import uuid  # unique ID
import time
from datetime import datetime


def get_timestamp():
    current_timestamp = datetime.now()
    return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")


# chat history file:
chat_history_dir = "data/chat_history/"


def main():  # streamlit run src/app/webpage.py --server.baseUrlPath=/d/code/LyricChat
    setup_config()  # setup webpage
    display_chat_history()  # display chat history
    chatbot = rag.LyricRAG()  # initialize RAG-LLM
    # start the conversation:
    user_input = st.chat_input(
        "Share what's on your mind. Wonda will find the perfect song to match your mood!"
    )
    if user_input:  # if user type something
        st.session_state.chat_history.append(human_msg(user_input))
        save_chat_history()
        with st.chat_message("Human"):
            st.markdown(user_input)
        # Stage one, do sentiment analysis for DB filtering
        progress_bar = st.progress(0, text="Identifying emotion...")
        chatbot.user_input = user_input
        chatbot.chain_classify()

        if chatbot.temp_response is not None:  # LLM suggest don't continue workflow
            with st.chat_message("AI"):
                progress_bar.progress(40, text="LLM suggest to stop workflow")
                st.markdown(chatbot.temp_response)
                model_response = chatbot.temp_response
        else:  # Stage two, song recommendation
            classified_result = f"Calssified emotion: **{chatbot.user_emotion}**"
            progress_bar.progress(40, text=classified_result)
            with st.chat_message("AI"):
                response = chatbot.chain_rag(top_r=1, stream=True)
                model_response = st.write_stream(rag.yield_stream(response))
                st_player(chatbot.youtube_link)
        # save chat history:
        st.session_state.chat_history.append(AI_msg(model_response))
        save_chat_history()  # save chat history


def setup_config():
    # TODO: set page configs
    st.set_page_config(page_title="LyricChat", page_icon="🎵", layout="centered")
    # st.title("LyricChat: Turn your Feelings into Melody")
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 32px; color: #333333;'>LyricChat: Turn your Feelings into Melody 🎼</h2>
        """,
        unsafe_allow_html=True,
    )
    # set colors:
    custom_css = """
    <style>
    .stApp {
        background-color: #E6F3FF;
    }
    .stButton>button {
        background-color: #4DA8DA;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
    }
    .stSelectbox>div>div>select {
        background-color: #FFFFFF;
    }
    .stHeader {
        background-color: #4DA8DA;
        color: white;
    }
    .element-container blockquote {
        background-color: #EAF4F9;  /* Softer, more muted blue for blockquotes */
        border-left: 5px solid #4DA8DA;  /* Blue left border */
        padding: 10px;
        margin: 10px 0;
    }
    .chat-message {
        background-color: #FFFFFF;  /* White background for chat messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    col1, col2 = st.columns(
        [1, 1], vertical_alignment="bottom"
    )  # Adjust column widths to be equal
    with col1:
        if st.button("Give us your feedback", use_container_width=True):
            js = "window.open('https://github.com/tctsung')"  # JavaScript to open link in new tab
            html = f"<script>{js}</script>"
            st.markdown(html, unsafe_allow_html=True)
    with col2:
        if st.button("Restart the Chat", use_container_width=True):
            restart_conversation()


def restart_conversation():
    """Helper for setup_confit() to restart the conversation"""
    if "session_ID" in st.session_state:
        del st.session_state.session_ID  # delete chat history
        del st.session_state.chat_history
    st.session_state.selected_artist = "All Artists"  # Reset to default artist


def display_chat_history():
    # TODO: load chat history into conversation
    if "session_ID" not in st.session_state:  # initialize a session w chat history
        st.session_state.chat_history = []
        st.session_state.session_ID = uuid.uuid4().hex
        # create an empty excel file for chat history:
        chat_history_file = (
            chat_history_dir + f"chat_history_{st.session_state.session_ID}.xlsx"
        )
        df_empty = pd.DataFrame(columns=["session_ID", "timestamp", "role", "content"])
        df_empty.to_excel(chat_history_file, engine="openpyxl", index=False)
    else:
        # show chat history on UI page:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                with st.chat_message("AI"):
                    st.markdown(msg["content"])
            elif msg["role"] == "user":
                with st.chat_message("Human"):
                    st.markdown(msg["content"])


def save_chat_history():
    if st.session_state.chat_history:
        # save chat history to excel file:
        msg = st.session_state.chat_history[-1]
        role = "Human" if msg["role"] == "user" else "AI"
        chat_history_file = (
            chat_history_dir + f"chat_history_{st.session_state.session_ID}.xlsx"
        )
        df_history = pd.read_excel(chat_history_file, engine="openpyxl")
        df_history.loc[df_history.shape[0]] = [
            st.session_state.session_ID,
            get_timestamp(),
            role,
            msg["content"],
        ]
        df_history.to_excel(chat_history_file, engine="openpyxl", index=False)


if __name__ == "__main__":
    main()
