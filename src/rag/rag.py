import os
# DB:
import json
from unidecode import unidecode
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
# from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# RAG:
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever


# Future work: 
# add chat history for LyricRAG
# add func: extract data from all folders at once
# turn create collection into -> create if not exist, append if have input_directory; don't append if same data
class SetupDB:
    """
    TODO: setup Qdrant vector DB for RAG 
    """
    embed_model = "BAAI/bge-base-en-v1.5"
    def __init__(self, input_directory=None, url_db = 'http://localhost:6333'):
        """
        param input_directory (str): directory to load data; if None, call the collection without adding docs
        """
        # set params:
        self.input_directory = input_directory
        self.url_db = url_db
        self.embeddings = FastEmbedEmbeddings(model_name = SetupDB.embed_model)

        # setup DB:
        self.create_vector_db()   
    def load_doc(self):
        docs = []                                             # buffer to save lanchain docs
        self.artist = get_artist_name(self.input_directory)   # metadata for parsing
        
        # read data from input directory:
        with open(os.path.join(self.input_directory, 'lyrics_filter.json'), 'r') as fp:
            lyrics = json.load(fp)
        with open(os.path.join(self.input_directory, 'lyrics_feature.json'), 'r') as fp:
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
        if self.input_directory is None:                   # call the collection without adding docs
            Qdrant_client = QdrantClient(url=self.url_db)
            self.vector_db = Qdrant(client=Qdrant_client, embeddings = self.embeddings, collection_name="lyrics")
        else:
            self.load_doc()                                # transform data from json to langchain Document 
            self.vector_db = Qdrant.from_documents(        # create collection
                self.langchain_docs,
                self.embeddings,
                url = self.url_db,
                collection_name="lyrics",
            )
class LyricRAG:
    sys_prompt = """You are Wonda, a deeply empathetic AI assistant. Your mission is to provide support, connect with users emotionally, 
and thoughtfully recommend a song that resonates with their current mood. 
Your priority is always the user's emotional well-being, providing comfort or inspiration as needed.

To achieve this:

1. Carefully analyze the user's input to determine the emotional tone—whether positive, neutral, or negative. Consider the context and specific language used to understand the user's feelings.
2. Based on the identified emotion, choose an appropriate response. Celebrate positive emotions or offer comfort and support for negative ones.
3. Use this emotional insight to select the most suitable song from the provided list, explaining the reasoning behind your choice.
"""
    rag_template = """Recommend one song from the followings options based only on the provided context:
<context>
{context}
</context>
"""
    formatting_instructions = """\nYour response should follow this structure:
1. Begin with a short paragraph that offers emotional support and connects with the user (up to 4 sentences).
2. Provide a brief description of the recommended song, explaining why it resonates with the user's current mood, but without mentioning its name or title (up to 2 sentences).
3. Share lyrics from the song that resonate with the user's current feelings (2 to 4 sentences).
4. Conclude with the song title and artist's name in the format of `— *<Title>* by <Artist>`
"""
    few_shots = """Format Example:
<example 1>
User: I'm feeling super duper awesome, oh yeahhhhh
Wonda: Wow, your excitement is absolutely infectious! It's so uplifting to see you in such high spirits—days like these remind us how incredible life can be when everything just clicks. Your positivity is a powerful force, and it's clear you're embracing every moment with joy.

To match your amazing vibe, I've picked a song that's all about celebrating life and feeling unstoppable. The lyrics will keep you riding that wave of happiness and remind you of your own strength and resilience.

"Started out with nothing at all
And even when I might fall down
I know my luck come back around."

— *I'm Happy* by Imagine Dragons
</example 1>

<example 2>
User: Sometimes I feel like giving up may be easier. But I also want fo fulfill my surrounding people expectation

Wonda: I can feel the weight you’re carrying—the push and pull between wanting to give up and striving to meet the expectations of those around you. It’s okay to feel overwhelmed, but remember that you don’t have to be perfect to be worthy of love and respect. You’re stronger than you think, and sometimes, it’s about giving yourself permission to take things one step at a time.

The song I’m sharing with you reflects those moments of self-doubt, yet it’s also a reminder that you’ve already proven yourself in so many ways. It encourages you to take it easy and trust that you’re enough, just as you are.

"Who made you think you weren't good enough?
Who made, who made, who made, who made you think that you weren't good enough?
Easy now. You don't have nothing left to prove
Easy now. Oh, it's laid out for you"

— *Easy* by Imagine Dragons
</example 2>

<example 3>
User: I hate my life

Wonda: 
I'm really sorry you're feeling this way. It's tough to face such overwhelming emotions, and it’s important to remember that reaching out for support can be a crucial step. You're not alone, and there are people who care about you and want to help.

In times of deep sadness and frustration, it's vital to find a source of understanding and solace. The song I’m recommending reflects the struggle of dealing with these intense feelings, and its lyrics offer a comforting reminder that it’s okay to feel this way and that change is possible.
"Can I wish on a star for another life?
'Cause it feels like I'm all on my own tonight
And I find myself in pieces"

— *My Life* by Imagine Dragons
</example 3>
"""
    def __init__(self, model = 'llama3',url_model = 'http://localhost:11434', url_db = 'http://localhost:6333'):
        # set params:
        self.model = model
        self.url_model = url_model
        self.url_db = url_db
        
        # load API:
        self.load_model()
        self.load_db()
        # buffers:
        # self.chat_history = []

    def load_model(self):
        num_ctx = 8192 if self.model == 'llama3' else 4096   # mistral only take 4096 as max context length
        self.llm = ChatOllama(model = self.model, temperature = 0.3, url = self.url_model, num_ctx=num_ctx)
    def load_db(self):
        Qdrant_client = QdrantClient(url=self.url_db)
        embeddings = FastEmbedEmbeddings(model_name = SetupDB.embed_model)
        self.vector_db = Qdrant(client=Qdrant_client, embeddings = embeddings, collection_name="lyrics")
    def rag_basic(self, user_input):
        combined_prompt = ChatPromptTemplate([
            ("system", LyricRAG.sys_prompt + LyricRAG.rag_template + LyricRAG.formatting_instructions + LyricRAG.few_shots),
            ("human", "{input}")             
            ])
        retriever=self.vector_db.as_retriever(search_kwargs={"k": 3})
        stuff_chain = create_stuff_documents_chain(self.llm, combined_prompt)   # chain to combine context and user input
        rag_chain = create_retrieval_chain(retriever, stuff_chain)
        self.rag_output = rag_chain.invoke({"input": user_input})    # do query
        self.display_msg()
    def display_msg(self):
        # TODO: print msg for testing purposes
        print("--------- Input ---------")
        print(self.rag_output['input'])
        print("--------- Model Output ---------")
        print(self.rag_output['answer'])
    def rag_conversation(self, user_input):
        combined_prompt = ChatPromptTemplate(
        [
            ("system", LyricRAG.sys_prompt + LyricRAG.rag_template + LyricRAG.formatting_instructions + LyricRAG.few_shots),
            # MessagesPlaceholder(variable_name="chat_history"),   # must use `chat_history`
            ("human", "{input}")                                # must use `input`
            # additional instruction for RAG:
            # ("human", "Given the above conversation, generate a search query to look up lyrics in order to recommend a different song")
        ])
    
    

######## Helper Functions ########
def get_artist_name(directory):
    # TODO: get artist name from LyricScraper.meta
    with open(os.path.join(directory, 'LyricScraper.meta'), 'r') as fp:
        for line in fp:
            if line.startswith('Artist name: '):
                return line.split(':')[1].strip()