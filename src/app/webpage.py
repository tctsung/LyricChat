from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
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
    # st.session_state.chat_history = [HumanMessage("Hi"), AIMessage("Yoooooooo")]    # for testing
    load_chat_history()   # load chat history
    chatbot = Chatbot()
    chatbot.chat()

def setup_config():
    # TODO: set page configs
    st.set_page_config(page_title="LyricChat", page_icon="🎵")
    st.title("LyricChat: Turn your Feelings into Melody")
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
    def __init__(self):
        self.rag_app = rag.LyricRAG()             # initialize RAG, remember to start ollama & qdrant server first
        
        # pre-load Ollama model:

        # create chains for sentiment analysis:
        self.rag_app.create_sentiment_chain()     # create chain for sentiment analysis


    def chat(self):
        # TODO: initialize conversation
        user_input = st.chat_input("Share what's on your mind. Wonda will find the perfect song to match your mood!")   # start chatting
        if user_input:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            with st.chat_message("Human"):
                st.markdown(user_input)

            with st.chat_message("AI"):
                # Stage one, do sentiment analysis -> for Advanced RAG metadata filtering
                sentiment, emotion = self.rag_app.sentiment_analysis(user_input)      
                sentiment = "Unidentified" if sentiment == "" else sentiment       # Replace empty strings with "Unidentified"
                emotion = "Unidentified" if emotion == "" else emotion
                classified_result = f"Calssified sentiment: **{sentiment}**, emotion: **{emotion}**"
                st.markdown(classified_result)
            with st.chat_message("AI"):
                ## Stage two, do advanced RAG
                model_response =st.write_stream(self.do_rag(user_input, emotion))   # stream the response

            # st.session_state.chat_history.append(AIMessage(content=classified_result))
            st.session_state.chat_history.append(AIMessage(content=model_response))
                
    def do_rag(self, user_input, emotion, memory=2):
        retriever = self.rag_app.create_customized_retriever(emotion)           # create retriever with filtered emotion
        rag_chain = self.rag_app.create_rag_conversation(customized_retriever = retriever)  # build chain
        
        for chunk in rag_chain.stream({"input": user_input,
                                       "chat_history": st.session_state.chat_history[-(2*memory):]}):
            if 'answer' in chunk:
                yield chunk['answer']





if __name__ == '__main__':
    main()