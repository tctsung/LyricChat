import pandas as pd
import os
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# load helper functions:
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
sys.path.append(src_dir)    # append src/ to sys.path to load helper.py     
from helper import *

# get secret tokens:
env_path = os.path.join(os.path.dirname(src_dir), ".env")  # load LyricChat/.env to env path
from dotenv import dotenv_values
ENV_VAR = dotenv_values(env_path)

# two kinds of DB building:
# 1. build at local url
# 2. build at remote url



# a class that turn input data to Langchain document
class BuildDB:
    embed_model = "BAAI/bge-base-en-v1.5"
    def __init__(self, collection_name, recreate = False, url_db = "http://localhost:6333"):
        """
        param collection_name (str): name of the collection in Qdrant
        param recreate (bool): if True, delete the collection & create a new one; 
                               if False, drop existing artists from metadata.csv files & add new data points
        url_db (str): Qdrant url, should be local path | remote Qdrant DB url
        """
        # check params:
        assert collection_name in ["genius", "open-lyrics"], "collection_name must be either 'genius' or 'open-lyrics'"
        
        # load input args:
        self.collection_name = collection_name
        self.url_db = url_db
        self.recreate = recreate
        self.embeddings = FastEmbedEmbeddings(model_name = BuildDB.embed_model)

        # prepare data:
        self.load_data()            # load input data (parquet file)
        self.recreate_or_filter()   # add metadata creation & filtering in the future
        self.df_2_langchain_doc() # turn pandas df to langchain doc

        # build Qdrant DB:

    def load_data(self):
        # read parquet file:
        if self.collection_name == "genius":
            # load latest parquet file:
            genius_latest_file = latest_file("data/genius/MERGED", file_extension=".parquet")
            self.df = pd.read_parquet(genius_latest_file)
            # load metadata:

        else:
            # add this for open-lyrics
            pass
    def recreate_or_filter(self):
        # TODO: delete old collection | filter input data
        self.Qdrant_client = QdrantClient(url=self.url_db)
        if self.recreate:      # delete collection
            self.Qdrant_client.delete_collection(collection_name=self.collection_name)
        else:
            # add this in the future -> create a metadata.csv, and avoid scrapping from the beginning
            pass
            # if self.collection_name == "genius":
            #     metadata_path = "data/genius/metadata.csv"
            #     metadata = pd.read_csv(metadata_path).astype(str)    # record of existing artist & timestamp
            # else:
            #     # add this for open-lyrics
            #     pass
    def df_2_langchain_doc(self):
        docs = []      # buffer to save lanchain docs
        for i, row in self.df.iterrows():
            docs.append(Document(
                page_content=row["lyrics"],
                id = row["id"],              # unique ID format: <artist name>|||<song title>|||<scrape order>|||<chunk order>
                metadata = {
                    'artist_name': row["artist_name"],
                    'song_title' : row["song_title"],
                    'parent_id': row["parent_id"],
                    'chunk_order' : row["chunk_order"],
                    'primary_emotion' : row["primary_emotion"], 
                    'secondary_emotion' : row["secondary_emotion"], 
                    'tertiary_emotion' : row["tertiary_emotion"],
                    'classification_result' : row["classification_result"]
                    }
                ))
        self.langchain_docs = docs
    def create_Qdrant_db(self):
        self.vector_db = QdrantVectorStore.from_documents(        # create collection
            self.langchain_docs,
            self.embeddings,
            url = self.url_db,
            collection_name=self.collection_name,
        )