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
chat_history_dir = "data\chat_history"


def main():  # streamlit run main.py
    setup_interface()  # setup webpage
    display_chat_history()  # display chat history
    chatbot = cache_LyricChat()  # initialize RAG-LLM
    # start the conversation:
    user_input = st.chat_input(
        "Share what's on your mind. Wonda will find the perfect song to match your mood!"
    )
    if user_input:  # if user type something
        with st.chat_message("Human"):
            st.markdown(user_input)
        # Stage one, do sentiment analysis for DB filtering
        progress_bar = st.progress(0, text="Identifying emotion...")
        chatbot.user_input = user_input
        chatbot.chat_history = st.session_state.chat_history
        chatbot.memory = 5
        chatbot.chain_classify()

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
                response = chatbot.chain_rag(top_r=5, stream=True)
                model_response = st.write_stream(rag.yield_stream(response))
                st_player(chatbot.youtube_link)
        # save chat history:
        st.session_state.chat_history.append(human_msg(user_input))  # save user input
        st.session_state.chat_history.append(AI_msg(model_response))
        if chatbot.youtube_link is not None:
            st.session_state.chat_history.append(video_msg(chatbot.youtube_link))
        st.rerun()  # update download button for chat history


@st.cache_resource  # Use cache_resource for class instances
def cache_LyricChat():
    return rag.LyricRAG()


def setup_interface():
    # TODO: set page configs
    # st.title("LyricChat: Turn your Feelings into Melody")
    st.set_page_config(page_title="LyricChat", page_icon="🎵", layout="centered")
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 32px; color: #333333;'>LyricChat: Turn your Feelings into Melody 🎼</h2>
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
    col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column widths to be equal
    with col1:
        st.link_button("Give us your feedback", "https://forms.gle/Xq2vo4TcVa4UMyXNA")
    with col3:
        if st.button("Restart the Chat", use_container_width=True):
            restart_conversation()
    with col2:
        df = pd.DataFrame(st.session_state.chat_history)
        if df.shape[0] > 0:
            output = BytesIO()
            # Write the DataFrame to the BytesIO object
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df.to_excel(
                    writer, index=False, sheet_name="Sheet1"
                )  # Save DataFrame to Excel
            st.download_button(
                label="Download Chat History",
                data=output.getvalue(),
                file_name=f"chat_history_{st.session_state.session_ID}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.button("Download Chat History", disabled=True, use_container_width=True)


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
