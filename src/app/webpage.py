from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd
import streamlit as st
from streamlit_player import st_player   # embedd music/video
import os
import re
import uuid    # unique ID
import time
from datetime import datetime
import sys
sys.path.append("src/rag/")
import rag
from importlib import reload
reload(rag)
from langchain_core.output_parsers import StrOutputParser

def get_timestamp():
	current_timestamp = datetime.now()
	return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
def clean_file_name(file_name):
	cleaned_name = re.sub(r'\s+', ' ', file_name.upper())
	cleaned_name = re.sub(r'[\s:-]', '_', cleaned_name)
	return cleaned_name

## Global variables:
# artists in DB:
artist_lst = pd.read_csv("data/qdrant/metadata.csv").artist.unique().tolist()
artist_lst.insert(0, "All Artists")
# chat history file:
chat_history_dir = "data/chat_history/"


def main():          # streamlit run src/app/webpage.py
    setup_config()   # setup basic info
    load_chat_history()   # load chat history
    artist = None if st.session_state.selected_artist == "All Artists" else st.session_state.selected_artist
    chatbot = Chatbot(artist)
    chatbot.chat()

def setup_config():
    # TODO: set page configs
    st.set_page_config(page_title="LyricChat", page_icon="🎵", layout="centered")
    # st.title("LyricChat: Turn your Feelings into Melody")
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 32px; color: #333333;'>LyricChat: Turn your Feelings into Melody 🎼</h2>
        """,
        unsafe_allow_html=True
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
    col1, col2 = st.columns([3, 1], vertical_alignment = "bottom")  # Create two columns for layout
    with col1:
        st.session_state.selected_artist = st.selectbox("Select an artist (or 'All Artists' for no filter):", artist_lst)
    with col2:
        if st.button("Restart the Chat", use_container_width=True):
            restart_conversation()
def restart_conversation():
    if "session_ID" in st.session_state:
        del st.session_state.session_ID                   # delete chat history
        del st.session_state.chat_history
    st.session_state.selected_artist = "All Artists"  # Reset to default artist
def load_chat_history():
    # TODO: load chat history into conversation
    if "session_ID" not in st.session_state:            # initialize a session w chat history
        st.session_state.chat_history = []
        st.session_state.session_ID = uuid.uuid4().hex
        # create an empty excel file for chat history:
        chat_history_file = chat_history_dir + f"chat_history_{st.session_state.session_ID}.xlsx"
        df_empty = pd.DataFrame(columns=["session_ID", "timestamp", "role", "content"])
        df_empty.to_excel(chat_history_file, engine="openpyxl", index=False)
    else:
        # show chat history on UI page:
        for msg in st.session_state.chat_history:
            if isinstance(msg, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(msg.content)
            elif isinstance(msg, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(msg.content)
def save_chat_history():
    if st.session_state.chat_history:
        # save chat history to excel file:
        msg = st.session_state.chat_history[-1]
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        chat_history_file = chat_history_dir + f"chat_history_{st.session_state.session_ID}.xlsx"
        df_history = pd.read_excel(chat_history_file, engine="openpyxl")
        df_history.loc[df_history.shape[0]] = [st.session_state.session_ID, get_timestamp(), role, msg.content]
        df_history.to_excel(chat_history_file, engine="openpyxl", index=False)
        
class Chatbot:
    def __init__(self, artist=None):
        self.rag_app = rag.LyricRAG()             # initialize RAG, remember to start ollama & qdrant server first
        self.artist = artist
        # pre-load Ollama model:

        # create chains for sentiment analysis:
        self.rag_app.create_sentiment_chain()     # create chain for sentiment analysis


    def chat(self, retry=3):
        # TODO: initialize conversation
        user_input = st.chat_input("Share what's on your mind. Wonda will find the perfect song to match your mood!")   # start chatting
        if user_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            save_chat_history()
            with st.chat_message("Human"):
                st.markdown(user_input)
            # Stage one, do sentiment analysis -> for Advanced RAG metadata filtering
            progress_bar = st.progress(0, text="Identifying sentiment and emotion...")
            while retry > 0:
                try:    # do classification
                    sentiment, emotion = self.rag_app.sentiment_analysis(user_input)
                    break
                except:
                    progress_bar = st.progress(10*(4-retry), text="Failed to classify sentiment and emotion... retrying...")
                    user_input = self.rag_app.rewrite_user_input(user_input)   # rewrite user input when can't get proper classification
                    retry -= 1
                    time.sleep(0.1)
            
            self.sentiment = "Unidentified" if sentiment == "" else sentiment       # Replace empty strings with "Unidentified"
            self.emotion = "Unidentified" if emotion == "" else emotion
            self.classified_result = f"Calssified sentiment: **{self.sentiment}**, emotion: **{self.emotion}**"
            progress_bar.progress(40, text=self.classified_result)
            with st.chat_message("AI"):
                ## Stage two, do advanced RAG
                model_response =st.write_stream(self.do_rag(user_input, progress_bar))    # stream the response
            ## Stage three, add the song link by Agent:
            # st_player("https://youtu.be/L_brIj-go8U?si=FaVmUTcI7h-pNr69")

            # save chat history:
            # st.session_state.chat_history.append(AIMessage(content=classified_result))
            st.session_state.chat_history.append(AIMessage(content=model_response))
            save_chat_history()   # save chat history
            
    def do_rag(self, user_input, progress_bar, memory=2):
        retriever = self.rag_app.create_customized_retriever(self.emotion, self.artist)           # create retriever with filtered emotion
        rag_chain = self.rag_app.create_rag_conversation(customized_retriever = retriever)  # build chain
        progress_bar.progress(60, text=self.classified_result)
        show_progress = True
        for chunk in rag_chain.stream({"input": user_input,
                                       "chat_history": st.session_state.chat_history[-(2*memory):]}):
            if show_progress:
                progress_bar.progress(80, text=self.classified_result)
            if 'answer' in chunk:
                if show_progress:
                    progress_bar.progress(100, text=self.classified_result)
                    show_progress = False
                yield chunk['answer']   # stream the output




if __name__ == '__main__':
    main()