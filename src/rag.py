from datetime import datetime
import random
import llm  # model
from llm import sys_msg, human_msg, AI_msg, agent_msg
import qdrant_db as qd  # database
import os
import rich  # for Markdown print at console
from rich.markdown import Markdown

# load prompts
import src.prompt as prompt

# structured output:
from pydantic import BaseModel, Field, model_validator, field_validator
from enum import Enum
from typing import Literal


# load API keys:
# from dotenv import dotenv_values
# ENV_VAR = dotenv_values(".streamlit\secrets.toml")


class LyricRAG:
    # include empty string for unidentified emotion
    backup_collection = "OpenLyrics"
    # prefer_collection = "NEFFEX"

    def __init__(
        self,
        deployment: Literal["cloud", "local"] = "cloud",
        memory=3,
    ):
        """
        TODO: main interface for RAG chatbot
        Args:
            deployment (str): cloud or local
            memory (int): no. of chat history to use (each memory is a full human & AI interaction)
        Attr:
            chat_history (list): user & chatbot conversation
            full_history (list): under the hood what model is actually doing
        """
        # set params:
        self.memory = memory
        self.language = "en"  # default language
        self.deployment = deployment
        if deployment == "cloud":
            self.base_url = None
            self.url_db = "https://94ddfbca-50be-4fb8-8791-ed716c146a08.europe-west3-0.gcp.cloud.qdrant.io:6333"
        else:
            self.base_url = "http://localhost:11434/"
            self.url_db = "http://localhost:6333"

        # load LLM model & DB:
        self.model = llm.InstructorLLM(
            deployment=self.deployment,
            base_url=self.base_url,
            GEMINI_API_KEY=os.environ["GEMINI_API_KEY"],
        )
        self.db = qd.QdrantVecDB(
            url_db=self.url_db, api_key=os.environ["Qdrant_API_KEY"]
        )
        # buffers:
        self.full_history = []  # full history of user input & LLM workflow
        self.chat_history = []  # chat history (only user input & chatbot response)

    def load_and_save_chat(
        self,
        input,
        msg_type: Literal["user", "assistant", "system", "agent"],
        language=None,
    ):
        """
        TODO: save user input/model output to chat history
        """
        # set language for this iter:
        if language:
            self.language = language
        # add current user_input into chat history:
        if msg_type == "user":
            self.user_input = input
            input = human_msg(input)
        else:
            # turn generator into str:
            if hasattr(input, "__iter__") and not isinstance(input, str):
                input = "".join(str(item) for item in input)
            input = {"role": msg_type, "content": input}
        self.chat_history.append(input)
        self.full_history.append(input)

    def stream_or_run(self, messages, stream):
        # TODO: helper for chains, return generator | string
        if stream:
            return self.model.stream(
                messages, chat_history=self.chat_history, memory=self.memory
            )
        # return response as whole string
        return self.model.run(
            messages, chat_history=self.chat_history, memory=self.memory
        )

    def chat(self, top_r=3):
        """
        TODO: chat interface at console, use chat_workflow() to yield results & save chat_history
        Args:
            top_r (int): number of songs to retrieve from DB as context
        Eg.
            chatbot = rag.LyricRAG()
            chatbot.chat()
        """
        while True:
            user_input = input("User input (or `exit` to leave): ")
            if user_input.lower() == "exit":
                break
            response = self.chat_workflow(user_input, top_r)
            markdown_text = Markdown(response)
            rich.print(user_input)
            rich.print(markdown_text)
            self.load_and_save_chat(response, msg_type="assistant")

    def chat_workflow(self, user_input, top_r=3):
        """
        TODO: Main logic workflow for the chatbot
        the streamlit chatbot workflow should be similar to this
        """
        # save user input for all chains:
        self.load_and_save_chat(input=user_input, msg_type="user")
        ## chain 1: identify user need:
        self.chain_classify()
        ## chain 2:
        # if emotional_support & song_recommendation are both False:
        if not (
            self.classification.emotional_support or self.classification.recommend_song
        ):
            return self.chain_problem_solving(stream=False)
        else:
            return self.chain_emotional_support(stream=False, top_r=top_r)

    def chain_classify(self, memory=3):
        """
        TODO: first workflow for the chatbot, classify if emotional_support & song_recommendation are needed
        return:
        """
        self.youtube_link = None  # reset youtube link
        messages = [sys_msg(prompt.sys_classify), human_msg(self.user_input)]
        classification = self.model.run(
            messages,
            schema=prompt.UserRequest,
            max_retries=5,
            chat_history=self.chat_history,
            memory=self.memory,
        )
        self.full_history.append(agent_msg(classification))
        self.classification = classification

    def chain_problem_solving(self, stream, language=None):
        if language is None:
            language = self.language
        sys_prompt = (
            prompt.sys_wonda + prompt.sys_problem_solving + prompt.languages[language]
        )
        messages = [sys_msg(sys_prompt), human_msg(self.user_input)]
        return self.stream_or_run(messages, stream=stream)

    def chain_emotional_support(self, stream, language=None, top_r=3):
        if language is None:
            language = self.language
        sys_prompt = (
            prompt.sys_wonda
            + prompt.languages[language]
            + prompt.psychotherapy_guidelines
        )
        if self.classification.recommend_song:
            self.chain_retrieve(top_r=top_r)  # retrieve songs at self.retrieved_context
            sys_prompt += (  # add sys instruction for song recommendation
                prompt.sys_recommend_song.format(context=self.retrieved_context)
                + prompt.one_shot
            )
        messages = [sys_msg(sys_prompt), human_msg(self.user_input)]
        return self.stream_or_run(messages, stream=stream)

    def chain_retrieve(self, top_r, user_emotion=None):
        """
        TODO: Retrive similar songs in string from DB filtered by user emotion
        Args:
            top_r: no. of points to retrieve for prefer_collection (backup_collection always retrieve k songs)
            use_backup: include backup_collection or not
        """

        # currently set should_conditions=None:
        should_conditions = (
            None
            if user_emotion is None
            else {
                "metadata.primary_emotion": self.user_emotion,
                "metadata.supporting_emotion": self.user_emotion,
            }
        )
        songs = self.db.read(
            collection_name=self.backup_collection,
            query=self.user_input,
            should_conditions=should_conditions,
            limit=top_r,
        )
        # # similarity search for backup DB:
        # songs_bu = self.db.read(
        #     collection_name=LyricRAG.backup_collection,
        #     query=self.user_input,
        #     should_conditions=should_conditions,
        #     limit=10,
        # )
        # songs.extend(songs_bu)
        self.retrieved_songs = songs
        self.chain_rerank()
        # turn to html format string for RAG
        self.retrieved_context = format_song(self.selected_song)
        self.full_history.append(agent_msg(self.retrieved_context))

    def chain_rerank(self):
        """TODO: rerank retrieved songs & collect required info for chatbot
        currently random, will be replaced by reranking algorithm
        """
        idx = random.randint(0, len(self.retrieved_songs) - 1)
        self.selected_song = self.retrieved_songs[idx]
        self.youtube_link = self.selected_song["metadata"]["youtube_link"]


######## Helper Functions ########
def yield_stream(chunks):
    """
    TODO: helper for InstructorLLM.stream to preprocess & yield the chunk
    this is for use with st.write_stream
    Eg.
    import src.llm as llm
    import src.rag.rag as rag
    import streamlit as st

    chatbot = rag.LyricRAG()
    response = chatbot.chat(user_input = "I'm feeling happy today")
    st.write_stream(rag.yield_stream(response))
    """
    for chunk in chunks:
        yield llm.preprocess_stream(chunk)


def get_timestamp():
    current_timestamp = datetime.now()
    return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_song(song_info):
    # TODO: turn vector DB results into html format LLM input
    # eg. ```<Artist name> Eminem <\Artist> <Song title> Rap God <\Song title> <Lyric> ... <\Lyric> ```
    metadata = song_info["metadata"]
    return f"""<Artist name> {metadata['artist']} </Artist> 
<Song title> {metadata['title']} </Song title> 
<Summary> {song_info['text']} </Summary>
<Lyric> {metadata['lyric'][:5000]} </Lyric>"""
