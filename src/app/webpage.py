from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
from streamlit_player import st_player   # embedd music/video
import os
import sys
sys.path.append("src/rag/")
import rag
from importlib import reload
reload(rag)
from langchain_core.output_parsers import StrOutputParser

# os.chdir("..")

def main():
    setup_config()   # setup basic info
    load_chat_history()   # load chat history
    artist = None if st.session_state.selected_artist == "All Artists" else st.session_state.selected_artist
    chatbot = Chatbot(artist)
    chatbot.chat()

def setup_config():
    # TODO: set page configs
    st.set_page_config(page_title="LyricChat", page_icon="🎵")
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
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # artist_options = ["All Artists", "NF", "Imagine Dragons", "Post Malone"]
    # st.session_state.selected_artist = st.selectbox(
    #     "Select an artist (or 'All Artists' for no filter):",
    #     artist_options
    # )
        # Add this near the end of the setup_config() function, after the artist selection
    col1, col2 = st.columns([3, 1], vertical_alignment = "bottom")  # Create two columns for layout
    with col1:
        artist_options = ["All Artists", "NF", "Imagine Dragons", "Post Malone"]
        st.session_state.selected_artist = st.selectbox(
            "Select an artist (or 'All Artists' for no filter):",
            range(len(artist_options)),
            format_func=lambda x: artist_options[x]
        )
        st.session_state.selected_artist = artist_options[st.session_state.selected_artist]
    
    with col2:
        if st.button("Restart the Chat", use_container_width=True):
            st.session_state.chat_history = []
def restart_conversation():
    st.session_state.chat_history = []
    st.session_state.selected_artist = "All Artists"  # Reset to default artist
    # Add any other variables you want to reset here
    st.experimental_rerun()  # This will rerun the script, effectively refreshing the page
def load_chat_history():
    # TODO: load chat history into conversation
    if "chat_history" not in st.session_state:    # initialize chat history
        st.session_state.chat_history = []
    else:
        for msg in st.session_state.chat_history:
            if isinstance(msg, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(msg.content)
            elif isinstance(msg, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(msg.content)

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
            
            self.sentiment = "Unidentified" if sentiment == "" else sentiment       # Replace empty strings with "Unidentified"
            self.emotion = "Unidentified" if emotion == "" else emotion
            self.classified_result = f"Calssified sentiment: **{self.sentiment}**, emotion: **{self.emotion}**"
            progress_bar.progress(40, text=self.classified_result)
            with st.chat_message("AI"):
                ## Stage two, do advanced RAG
                model_response =st.write_stream(self.do_rag(user_input, progress_bar))    # stream the response
            # st.session_state.chat_history.append(AIMessage(content=classified_result))
            st.session_state.chat_history.append(AIMessage(content=model_response))
            
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