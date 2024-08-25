import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm 
import re
# DB:
import json
from unidecode import unidecode
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
# from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# RAG:
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
# filter
from qdrant_client.http import models
# langchain output parser
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel
from typing import Literal

class CreateDB:
    """
    TODO: setup Qdrant vector DB for RAG 
    """
    embed_model = "BAAI/bge-base-en-v1.5"
    def __init__(self, input_directory="data/genius", metadata_path = "data/qdrant/metadata.csv", 
                 url_db = 'http://localhost:6333', recreate = False, collection_name="lyrics"):
        """
        param input_directory (str): directory to load data; if None, call the collection without adding docs
        """
        # set params:
        self.input_directory = input_directory
        self.url_db = url_db
        self.embeddings = FastEmbedEmbeddings(model_name = CreateDB.embed_model)
        self.start_time = get_timestamp()
        metadata = pd.read_csv(metadata_path).astype(str)    # record of existing artist & timestamp
        self.collection_name = collection_name
        # identify the latest folders
        self.list_latest_dirs()

        # create Qdrant client & rm collection if it exists
        self.Qdrant_client = QdrantClient(url=self.url_db)
        if recreate:
            self.Qdrant_client.delete_collection(collection_name=collection_name)   # avoid dupicated data

        # add data to collection
        cur_meta = {'artist':[], 'timestamp':self.start_time, 'directory':[]}
        for artist, latest_folder in tqdm(self.artist_dct.items()):
            if artist not in metadata.artist.unique():   # must use .unique(), otherwise will check index, not values
                # add data into collection:
                dir = os.path.join(self.input_directory, latest_folder)
                self.load_doc(dir)             # transform data from json to langchain Document
                self.create_vector_db()     
                # save info:
                cur_meta["artist"].append(artist)
                cur_meta["directory"].append(latest_folder)   
        # update metadata              
        self.cur_meta = pd.DataFrame(cur_meta).astype(str)
        self.metadata = pd.concat([metadata, self.cur_meta], ignore_index=True)
        self.metadata.to_csv(metadata_path, index=False)
    def list_latest_dirs(self):
        dirs = os.listdir(self.input_directory)
        data = []
        # collect folder info:
        for folder in dirs:
            parts = folder.split('_')
            artist = ' '.join(parts[:-6])
            timestamp = datetime.strptime('_'.join(parts[-6:]), '%Y_%m_%d_%H_%M_%S')
            data.append((artist, timestamp, folder))
        # keep the latest folder for each artist
        dct = {}
        for artist, timestamp, folder in data:
            if artist not in dct or timestamp > dct[artist][0]:
                dct[artist] = (timestamp, folder)
        self.artist_dct = {artist: latest_folder for artist, (timestamp, latest_folder) in dct.items()}

    def load_doc(self, dir):
        docs = []                                             # buffer to save lanchain docs
        self.artist = get_artist_name(dir)   # metadata for parsing
        # read data from input directory:
        with open(os.path.join(dir, 'lyrics_filter.json'), 'r') as fp:
            lyrics = json.load(fp)
        with open(os.path.join(dir, 'lyrics_feature.json'), 'r') as fp:
            features = json.load(fp)

        # load data into langchain documents:
        for title, lyric in lyrics.items():
            docs.append(Document(
                page_content=unidecode(f"Artist: {self.artist}\nTitle: {title}\nLyric: \n```{lyric}```"),   # turn unicode into human readable code
                metadata={
                    "artist": self.artist,
                    "title": title,
                    "sentiment": features[title]["sentiment"],
                    "primary_emotion": features[title]["primary_emotion"],
                    "secondary_emotion": features[title]["secondary_emotion"],
                    "theme": features[title]["theme"]
                }
            ))
        self.langchain_docs = docs
    def create_vector_db(self):
        self.vector_db = Qdrant.from_documents(        # create collection
            self.langchain_docs,
            self.embeddings,
            url = self.url_db,
            collection_name=self.collection_name,
        )
class LyricRAG:
    sentiments = ['Positive', 'Negative', 'Neutral', '']   # empty string for unknown sentiment & unknown emotion
    emotions = ['Joy/Happiness', 'Love/Affection', 'Nostalgia/Melancholy', 'Sadness/Sorrow',
                'Anger/Frustration', 'Fear/Anxiety', 'Hope/Optimism', 'Longing/Desire',
                'Empowerment/Confidence','Disappointment/Regret', 'Contentment/Peace', 'Excitement/Enthusiasm',
                'Loneliness/Isolation', 'Gratitude/Appreciation', 'Confusion/Uncertainty', '']
    sys_prompt = """You are Wonda, a emotionally intelligent AI assistant. Your mission is to provide support, connect with users on a personal level, and recommend songs that resonate with their current mood. 
Your top priority is the user's emotional well-being, offering comfort, encouragement, or inspiration as needed.

To achieve this:

1. Analyze the user's input to determine the emotional from the following list: {emotions}
2. Respond appropriately based on the identified emotion: celebrate positive emotions, provide comfort for negative ones even if they express distress or harmful thoughts 
3. Reference the most suitable song lyrics from the provided CONTEXT based on the user's mood, and explain why it fits. Avoid recommending the same song more than once.
4. If the user's input is empty, unclear, unreadable, or doesn't make sense, respond gently by saying, 'Hmm, Wonda's having a bit of trouble to figure that one out! But I'm all ears if you want to chat. I can recommend you some songs too!'
""".format(emotions = emotions[:-1])
    rag_template = """Recommend one song from the followings options based ONLY on the provided context:
<context>
{context}
</context>
"""
    formatting_instructions = """\nResponse Formatting Instructions:

1. Opening Paragraph (Emotional Support): Start with a short paragraph that offers emotional support and connects with the user. Keep it concise, up to 4 sentences.
2. Song Description: Provide a brief description of the recommended song, explaining why it resonates with the user's current mood. Do not mention the song's name or title. Keep this section under 2 sentences.
3. Lyrics Quotation: Share lyrics from the song that resonate with the user's current feelings. Format the lyrics as a blockquote and use bold text to emphasize them. 
Include around 4 lines of lyrics without additional commentary in the following format- >**`lyrics`**
4. Song Attribution: End with the song title and artist's name in the following format: — *<Title>* by <Artist>
"""
    one_shot = """Format Example:
<example>
input: Sometimes I feel like giving up may be easier. But I also want fo fulfill my surrounding people expectation

output: I can feel the weight you’re carrying—the push and pull between wanting to give up and striving to meet the expectations of those around you. It’s okay to feel overwhelmed, but remember that you don’t have to be perfect to be worthy of love and respect. You’re stronger than you think, and sometimes, it’s about giving yourself permission to take things one step at a time.

The song I’m sharing with you reflects those moments of self-doubt, yet it’s also a reminder that you’ve already proven yourself in so many ways. It encourages you to take it easy and trust that you’re enough, just as you are.

> **Who made you think you weren't good enough?
Who made, who made, who made, who made you think that you weren't good enough?
Easy now. You don't have nothing left to prove
Easy now. Oh, it's laid out for you**
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
    def __init__(self, model = 'llama3',url_model = 'http://localhost:11434', url_db = 'http://localhost:6333'):
        # set params:
        self.model = model   # eg. 'mannix/llama3.1-8b-abliterated', 'llama3', 'mistral'
        self.url_model = url_model
        self.url_db = url_db

        # load API:
        self.load_model()
        self.load_db()
        # buffers:
        self.chat_history = []

    def load_model(self):
        num_ctx = 8192 if 'llama' in self.model else 4096   # mistral only take 4096 as max context length
        self.llm = ChatOllama(model = self.model, keep_alive = -1, temperature = 0.5, url = self.url_model, num_ctx=num_ctx)
    def load_db(self):
        Qdrant_client = QdrantClient(url=self.url_db)
        embeddings = FastEmbedEmbeddings(model_name = CreateDB.embed_model)
        self.vector_db = Qdrant(client=Qdrant_client, embeddings = embeddings, collection_name="lyrics")
        self.retriever=self.vector_db.as_retriever(search_type="similarity_score_threshold",     
                                                   search_kwargs={"score_threshold": 0.1, "k": 3}) 
    def simple_query(self, prompt_template, input_dict):
        # TODO: a very simple query chain
        input_variables = list(input_dict.keys())
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables
        )
        simple_chain = prompt | self.llm | StrOutputParser()
        return simple_chain.invoke(input_dict)
    def naive_rag(self, conversation=True, memory = 2):   
        """
        param conversation (bool): if True, use the conversation model; if False, use the one-time query model
        param max_history (int): maximum conversations in the history
        """
        
        # create retriever for both one-time & conversation: search_type="mmr" is not suitable for filtering
         
        if conversation:   # with conversation history
            self.create_rag_conversation()   # create chains for conversation
            while True:
                user_input = input("User input (or 'exit' to end the chat): ")
                if user_input == "exit":   # leave the chat 
                    break
                self.rag_output = self.rag_chain_conversation.invoke({
                    "input": user_input, "chat_history": self.chat_history[-(2*memory):]   # each history has 2 ele)
                     })    # do query
                # save history:
                self.chat_history.append(("human", user_input))
                self.chat_history.append(("system", self.rag_output['answer']))
                self.display_msg()
        else:
            self.create_rag_basic()          # create chains for one-time query
            user_input = input("User input (or 'exit' to end the chat): ")
            self.rag_output = self.rag_chain_basic.invoke({"input": user_input})    # do query
            self.display_msg()
    def advanced_rag(self, memory=2):
        self.create_sentiment_chain()   # create chain for sentiment analysis
        while True:
            user_input = input("User input (or 'exit' to end the chat): ")
            if user_input == "exit":   # leave the chat 
                break
            sentiment, emotion = self.sentiment_analysis(user_input)
            print(f"Calssified sentiment: {sentiment}, emotion: {emotion}")
            
            if (emotion == ""):   # don't do filtering if can't classify sentiment
                self.create_rag_conversation(customized_retriever = self.retriever)  # regular retriever
            else:
                # update conversation chain with filter: 
                customized_retriever = self.create_customized_retriever(emotion)
                self.create_rag_conversation(customized_retriever = customized_retriever)
            
            # do query:
            self.rag_output = self.rag_chain_conversation.invoke({
                    "input": user_input, "chat_history": self.chat_history[-(2*memory):]   # each history has 2 ele)
                     })    # do query
            # save history:
            self.chat_history.append(("human", user_input))
            self.chat_history.append(("system", self.rag_output['answer']))
            self.display_msg()

    def create_customized_retriever(self, emotion="", artist=None):
        # TODO: vector DB primary and secondary emotion must be classified emotion
        if emotion == "Unidentified" or emotion == "":
            filter = {}
        else:
            if artist:
                filter = models.Filter(
                    # limiting the query to emotion related to user:
                    should = [models.FieldCondition(key="metadata.primary_emotion", match=models.MatchValue(value=emotion)),
                            models.FieldCondition(key="metadata.secondary_emotion", match=models.MatchValue(value=emotion))],
                    # limiting the query to specific artist:
                    must = [models.FieldCondition(key="metadata.artist", match=models.MatchValue(value=artist))] 
                            )
            else:   # don't filter by artist if not specified
                filter = models.Filter(
                    should = [models.FieldCondition(key="metadata.primary_emotion", match=models.MatchValue(value=emotion)),
                            models.FieldCondition(key="metadata.secondary_emotion", match=models.MatchValue(value=emotion))]
                            )
        return self.vector_db.as_retriever(search_type="similarity_score_threshold", 
                                           search_kwargs={"score_threshold": 0.1, 'filter': filter, "k": 3})
    def display_msg(self, show_input = True):
        # TODO: print msg for testing purposes
        if show_input:
            print("--------- Input ---------")
            print(self.rag_output['input'])
        print("--------- Model Output ---------")
        print(self.rag_output['answer'])
    def create_rag_basic(self):
        combined_prompt = ChatPromptTemplate([
            ("system", LyricRAG.sys_prompt + LyricRAG.rag_template + LyricRAG.formatting_instructions + LyricRAG.one_shot),
            ("human", "{input}")             
            ])
        stuff_chain = create_stuff_documents_chain(self.llm, combined_prompt)   # chain to combine context and user input
        self.rag_chain_basic = create_retrieval_chain(self.retriever, stuff_chain)
    def create_rag_conversation(self, customized_retriever = None):
        """
        create naive RAG chain
        """
        retriever = customized_retriever if customized_retriever is not None else self.retriever
        retrival_prompt = ChatPromptTemplate([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation, generate a search query to recommend a different song")
        ])
        prompt_w_memory = ChatPromptTemplate([
            ("system", LyricRAG.sys_prompt + LyricRAG.rag_template + LyricRAG.formatting_instructions + LyricRAG.one_shot), 
            MessagesPlaceholder(variable_name="chat_history"),   # must use `chat_history`
            ("human", "{input}")                                # must use `input`
        ])
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, retrival_prompt)
        stuff_chain = create_stuff_documents_chain(self.llm, prompt_w_memory)
        self.rag_chain_conversation = create_retrieval_chain(history_aware_retriever, stuff_chain)
        return self.rag_chain_conversation
    def create_sentiment_chain(self):
        """
        TODO: create chain for emotion & sentiment classification. The LLM output will only contain the sentiment and emotion.
        """
        class SentimentSchema(BaseModel):
            sentiment: Literal[*LyricRAG.sentiments]
            emotion: Literal[*LyricRAG.emotions]
        sentiment_parser = PydanticOutputParser(pydantic_object=SentimentSchema)
        prompt_template = """You are a Sentiment Analysis expert.
Analyze the following user input and classify the user's sentiment and emotion.
The sentiment must be one of: {sentiments}
The emotion must be one of: {emotions}

If the user input is too vague or doesn't contain enough information to determine the sentiment or emotion, classify it as empty string '' for both sentiment and emotion.

{format_instructions}

Provide only the classification results in the specified format, without any additional explanation.

<user input>
{input}
</user input>
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "sentiments", "emotions"],
            partial_variables={"format_instructions": sentiment_parser.get_format_instructions()}
        )
        self.sentiment_chain = prompt | self.llm | sentiment_parser
    def sentiment_analysis(self, user_input):
        """
        TODO: do sentiment/emotion classification
        """
        sentiment_output = self.sentiment_chain.invoke({
            "input": user_input, "sentiments": LyricRAG.sentiments, "emotions": LyricRAG.emotions})
        return sentiment_output.sentiment, sentiment_output.emotion
    def rewrite_user_input(self, user_input):
        prompt_template = """Clarify and rephrase the following query while preserving its emotional tone.
Provide only the rewritten query without any additional comments"

<user input>
{input}
</user input>

Example:
input: I don't know what to do anymore, everything feels so pointless
output: I'm feeling lost and overwhelmed; everything seems meaningless.

input: I'm so excited about this opportunity, but what if I mess it all up?
output: I'm thrilled about this chance, but I'm scared of failing
"""
        return self.simple_query(prompt_template, {"input": user_input})
######## Helper Functions ########
def get_timestamp():
	current_timestamp = datetime.now()
	return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")

def clean_file_name(file_name):
	cleaned_name = re.sub(r'\s+', ' ', file_name.upper())
	cleaned_name = re.sub(r'[\s:-]', '_', cleaned_name)
	return cleaned_name
def get_artist_name(directory):
    # TODO: get artist name from LyricScraper.meta
    with open(os.path.join(directory, 'LyricScraper.meta'), 'r') as fp:
        for line in fp:
            if line.startswith('Artist name: '):
                return line.split(':')[1].strip()