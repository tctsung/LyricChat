from datetime import datetime
import random
import llm  # model
from llm import sys_msg, human_msg, AI_msg
import qdrant_db as qd  # database
import os

# structured output:
from pydantic import BaseModel, Field, model_validator, field_validator
from enum import Enum
from typing import Literal


# load API keys:
# from dotenv import dotenv_values
# ENV_VAR = dotenv_values(".streamlit\secrets.toml")

# ReAct prompt:


# Instructured output schema:
class Emotions(str, Enum):
    Joy = "Joy"
    Love = "Love"
    Nostalgia = "Nostalgia"
    Sadness = "Sadness"
    Anger = "Anger"
    Fear = "Fear"
    Hope = "Hope"
    Desire = "Desire"
    Confidence = "Confidence"
    Regret = "Regret"
    Peace = "Peace"
    Excitement = "Excitement"
    Loneliness = "Loneliness"
    Gratitude = "Gratitude"
    Confusion = "Confusion"
    Betrayal = "Betrayal"
    Ambition = "Ambition"
    Forgiveness = "Forgiveness"
    Freedom = "Freedom"
    Unidentified = "Unidentified"


class UserEmotion(BaseModel):
    """
    TODO: Schema for user emotion classification in chat
    use `recommendation_status` to decide whether to continue workflow
    """

    human_readable: bool = Field(
        description="Boolean judgment indicating whether the user input is human readable"
    )
    recommendation_status: Literal["related", "unrelated"] = Field(
        description="Indicates whether the user's input is related or unrelated to song recommendation. "
        "Set to 'related' if the input provides enough emotional or contextual information for a recommendation, "
        "otherwise 'unrelated' to prompt the user for more input."
    )
    emotions: list[Emotions] = Field(
        description="The top two emotions observed in the lyrics. Two emotions must be different unless it's unidentified."
    )
    response: str = Field(
        description="Brief and helpful Chatbot response to user input. Encourage user to chat more or share their feelings for song recommendations."
    )

    @field_validator("emotions")
    def Emotions_are_diff(cls, val):
        # check if there are exactly two emotions and they are different
        if len(val) != 2:
            raise ValueError("There must be two emotions")
        return [c.value for c in val]  # turn ENUM to str after validation


class LyricRAG:
    # include empty string for unidentified emotion
    emotions = [e.value for e in Emotions]
    # system prompt for emotional classification (first step)
    sys_prompt_classify = """You are Wonda, a emotionally intelligent AI assistant. Analyze the user's input to determine the emotional context and sentiment.
If the input relates to song recommendation, simply do emotion classification.  
If the input is unrelated to song recommendation, try your best to provide brief and helpful response, but remind the user that you're here to recommend songs based on their mood.  
If the input is empty or not human readable, encourage the user to chat more or share their feelings to receive song recommendations.
"""
    # system prompt for chatbot output (step 2 in workflow)
    sys_prompt_chat = """You are Wonda, a emotionally intelligent AI assistant. Your mission is to provide support, connect with users on a personal level, and recommend songs that resonate with their current mood. 
Your top priority is the user's emotional well-being, offering comfort, encouragement, or inspiration as needed."""
    unclear_response = """If the user's input is empty, unclear, unreadable, or doesn't make sense, respond gently by saying, `Hmm, Wonda's having a bit of trouble to figure that one out!
But I'm all ears if you want to chat. I can recommend you some songs too!`"""
    # system prompt for Chatbot output (all steps in one prompt)
    sys_prompt_ReACT = """You are Wonda, a emotionally intelligent AI assistant. Your mission is to provide support, connect with users on a personal level, and recommend songs that resonate with their current mood. 
Your top priority is the user's emotional well-being, offering comfort, encouragement, or inspiration as needed.

To achieve this:

1. Analyze the user's input to determine the emotional context and sentiment.
2. Respond appropriately based on the identified emotion: celebrate positive emotions, provide comfort for negative ones even if they express distress or harmful thoughts 
3. Reference the most suitable song lyrics from the provided CONTEXT based on the user's mood, and explain why it fits. Avoid recommending the same song more than once.

Recommend one song from the followings options based ONLY on the provided context:
<context>
{context}
</context>

Response Formatting Instructions:

1. Opening Paragraph (Emotional Support): Start with a short paragraph that offers emotional support and connects with the user. Keep it concise, up to 4 sentences.
2. Song Description: Provide a brief description of the recommended song, explaining why it resonates with the user's current mood. Do not mention the song's name or title. Keep this section under 2 sentences.
3. Lyrics Quotation: Share lyrics from the song that resonate with the user's current feelings. Format the lyrics as a blockquote and use bold text to emphasize them. 
Include around 4 lines of lyrics without additional commentary. IMPORTANT: Add two spaces at the end of each line (except the last line) to create line breaks:

>**lyric line 1**  [two spaces here]
>**lyric line 2**  [two spaces here]
>**lyric line 3**  [two spaces here]
>**lyric line 4**
4. Song Attribution: End with the song title and artist's name in the following format: — *<Title>* by <Artist>
"""
    one_shot = """Format Example:
<example>
input: Sometimes I feel like giving up may be easier. But I also want fo fulfill my surrounding people expectation

output: I can feel the weight you’re carrying—the push and pull between wanting to give up and striving to meet the expectations of those around you. It’s okay to feel overwhelmed, but remember that you don’t have to be perfect to be worthy of love and respect. You’re stronger than you think, and sometimes, it’s about giving yourself permission to take things one step at a time.

The song I’m sharing with you reflects those moments of self-doubt, yet it’s also a reminder that you’ve already proven yourself in so many ways. It encourages you to take it easy and trust that you’re enough, just as you are.

> **Who made you think you weren't good enough?**  
> **Who made, who made, who made, who made you think that you weren't good enough?**  
> **Easy now. You don't have nothing left to prove**  
> **Easy now. Oh, it's laid out for you**  
\n
— *Easy* by Imagine Dragons
</example>
    
"""
    few_shots = """Format Example:
<example 1>
input: I'm feeling super duper awesome, oh yeahhhhh
output: Wow, your excitement is absolutely infectious! It's so uplifting to see you in such high spirits—days like these remind us how incredible life can be when everything just clicks. Your positivity is a powerful force, and it's clear you're embracing every moment with joy.

To match your amazing vibe, I've picked a song that's all about celebrating life and feeling unstoppable. The lyrics will keep you riding that wave of happiness and remind you of your own strength and resilience.

> **Started out with nothing at all
And even when I might fall down
I know my luck come back around.**

— *I'm Happy* by Imagine Dragons
</example 1>

<example 2>
input: Sometimes I feel like giving up may be easier. But I also want fo fulfill my surrounding people expectation

output: I can feel the weight you’re carrying—the push and pull between wanting to give up and striving to meet the expectations of those around you. It’s okay to feel overwhelmed, but remember that you don’t have to be perfect to be worthy of love and respect. You’re stronger than you think, and sometimes, it’s about giving yourself permission to take things one step at a time.

The song I’m sharing with you reflects those moments of self-doubt, yet it’s also a reminder that you’ve already proven yourself in so many ways. It encourages you to take it easy and trust that you’re enough, just as you are.

> **Who made you think you weren't good enough?
Who made, who made, who made, who made you think that you weren't good enough?
Easy now. You don't have nothing left to prove
Easy now. Oh, it's laid out for you**

— *Easy* by Imagine Dragons
</example 2>

<example 3>
input: I hate my life

output: 
I'm really sorry you're feeling this way. It's tough to face such overwhelming emotions, and it’s important to remember that reaching out for support can be a crucial step. You're not alone, and there are people who care about you and want to help.

In times of deep sadness and frustration, it's vital to find a source of understanding and solace. The song I’m recommending reflects the struggle of dealing with these intense feelings, and its lyrics offer a comforting reminder that it’s okay to feel this way and that change is possible.
> **Can I wish on a star for another life?
'Cause it feels like I'm all on my own tonight
And I find myself in pieces**

— *My Life* by Imagine Dragons
</example 3>
"""
    sys_prompt_rewrite = """Clarify and rephrase the following query while preserving its emotional tone.
Provide only the rewritten query without any additional comments"

Example:
input: I don't know what to do anymore, everything feels so pointless
output: I'm feeling lost and overwhelmed; everything seems meaningless.

input: I'm so excited about this opportunity, but what if I mess it all up?
output: I'm thrilled about this chance, but I'm scared of failing

<user input>
{input}
</user input>
"""

    def __init__(
        self,
        deployment: Literal["cloud", "local"] = "cloud",
        url_db=None,
        collection_name="NEFFEX",
    ):
        """
        TODO: main interface for RAG chatbot
        Args:
            deployment (str): cloud or local
            url_db (str): url for Qdrant DB
            collection_name (str): collection name to use (should be list in the future)
        """
        # set params:
        self.deployment = deployment
        self.collection_name = collection_name
        if deployment == "cloud":
            self.base_url = None
            self.url_db = (
                url_db
                if url_db
                else "https://94ddfbca-50be-4fb8-8791-ed716c146a08.europe-west3-0.gcp.cloud.qdrant.io:6333"
            )
        else:
            self.base_url = "http://localhost:11434/"
            self.url_db = url_db if url_db else "http://localhost:6333"

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
        self.chat_history = []

    def chat(self, user_input, chat_history: list = [], memory=0, top_r=5):
        """
        TODO: chain_classify +  chain_rag
        Args:
            user_input (str): user input for chatbot
            stream (bool): if True, will return a generator
            top_r (int): number of songs to retrieve from DB as context
        Attributes:
            user_emotion (list): emotions classified from user input
            classify_res (UserEmotion): classification result
            retrieved_context (str): retrieved songs in html format

        Eg.
            from IPython.display import Markdown, display
            chatbot = rag.LyricRAG()
            response = chatbot.chat("I'm feeling happy", stream=False)
            display(Markdown(response))
        """
        self.user_input = user_input  # save user input for all chains
        self.chat_history = chat_history  # save chat history
        self.memory = memory  # save memory for chat history
        # chain 1: classify user emotion
        self.chain_classify()
        if self.temp_response:
            return self.temp_response
        # chain 2: RAG for song recommendation
        return self.chain_rag(
            top_r, stream=False
        )  # return response as string or generator

    def chain_classify(self):
        """
        TODO: first workflow for the chatbot, classify user input emotion
        return: primary_emotion, supporting_emotion
        """
        self.temp_response = None  # buffer for temp response
        messages = [sys_msg(LyricRAG.sys_prompt_classify), human_msg(self.user_input)]
        res = self.model.run(
            messages,
            schema=UserEmotion,
            max_retries=5,
            chat_history=self.chat_history,
            memory=self.memory,
        )
        self.user_emotion = [
            x for x in res.emotions if x != "Unidentified"
        ]  # for DB filtering
        self.classify_res = res

        if (res.recommendation_status == "unrelated") or (res.human_readable == False):
            # user input is unclear/unrelated, suggest to chat more
            self.temp_response = res.response  # update temp response

    def chain_retrieve(self, top_r):
        """
        TODO: Retrive similar songs in string from DB filtered by user emotion
        """

        # Set should_conditions based on emotions without 'Unidentified':
        should_conditions = (
            None
            if not self.user_emotion
            else {
                "metadata.primary_emotion": self.user_emotion,
                "metadata.supporting_emotion": self.user_emotion,
            }
        )

        # similarity search:
        songs = self.db.read(
            collection_name=self.collection_name,
            query=self.user_input,
            should_conditions=should_conditions,
            limit=top_r,
        )
        self.retrieved_songs = songs
        self.chain_rerank()
        # turn to html format string for RAG
        self.retrieved_context = format_song(self.selected_song)

    def chain_rag(self, top_r=5, stream=True):
        """
        TODO: RAG for song recommendation; combine retrieved songs with output template
        Args:
            streamlit (bool): if True, will return a generator for st.write_stream;
                              otherwise return the whole response as string
        return: LLM response as string or generator
        """
        self.chain_retrieve(top_r)  # get retrieved songs at self.retrieved_context
        messages = [
            sys_msg(
                LyricRAG.sys_prompt_ReACT.format(context=self.retrieved_context)
                + LyricRAG.one_shot
            ),
            human_msg(self.user_input),
            AI_msg(f"User emotion: {self.user_emotion}"),
        ]
        if stream:
            return self.model.stream(messages)  # return generator
        return self.model.run(
            messages, chat_history=self.chat_history, memory=self.memory
        )  # return response as whole string

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
    return f"""<Artist name> {metadata['artist']} <\Artist> 
<Song title> {metadata['title']} <\Song title> 
<Summary> {song_info['text']} <\Summary>
<Lyric> {metadata['lyric'][:5000]} <\Lyric>"""
