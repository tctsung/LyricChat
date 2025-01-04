# env
# pip install qdrant-client pyarrow sentence-transformers[onnx-gpu]

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny  # filter
import types  # for generator type
from typing import Literal
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import dotenv_values

ENV_VAR = dotenv_values(".streamlit\secrets.toml")


class QdrantVecDB:
    def __init__(
        self,
        model="models/BAAI_bge-small-en-v1.5",
        device="cpu",
        url_db="http://localhost:6333",
        api_key=None,
    ):
        """
        TODO: customized Qdrant interface for LyricChat, includes CRUD operations
        """
        # setup args:
        self.model = model
        self.device = device
        self.url_db = url_db
        if api_key is None:  # get from .env if not provided
            api_key = ENV_VAR.get("Qdrant_API_KEY", None)

        # setup model & DB client:
        self.load_model()  # load embedding model
        self.client = QdrantClient(url=self.url_db, api_key=api_key)

    def load_model(self, model=None):
        """TODO: load the sentence transformer model & get the embedding dimension"""
        # change to new model if provided:
        if model is not None:
            self.model = model
        # load model:
        self.embedding_model = SentenceTransformer(self.model)
        self.embedding_model.to(self.device)  # move to GPU if available

        # get embedding dimension:
        temp_output = self.embedding_model.encode("", batch_size=1)
        self.embedding_dimension = temp_output.shape[0]

    def read(
        self,
        collection_name,
        query: str,
        limit: int = 5,
        should_conditions: dict = None,
        must_conditions: dict = None,
    ):
        """
        TODO: similarity search with query
        """
        # create filter:
        filter = self._create_qdrant_filter(should_conditions, must_conditions)

        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=self.embedding_model.encode(query),
            with_payload=True,
            limit=limit,
            query_filter=filter,
        )
        return [result.payload for result in search_result]

    def _create_qdrant_filter(self, should_conditions=None, must_conditions=None):
        """
        TODO: Helper for read(); create Qdrant filter from should & must conditions
        Eg. should_conditions = {"metadata.primary_emotion": "happy", "metadata.supporting_emotion": ["sad", "angry"]}
        """

        def create_field_condition(key, value):
            # use MatchAny for list of values, MatchValue for single value
            if isinstance(value, (list, tuple, types.GeneratorType)):
                return FieldCondition(key=key, match=MatchAny(any=value))
            else:
                return FieldCondition(key=key, match=MatchValue(value=value))

        should_filters, must_filters = [], []
        if should_conditions:
            should_filters = [
                create_field_condition(key, value)
                for key, value in should_conditions.items()
            ]

        if must_conditions:
            must_filters = [
                create_field_condition(key, value)
                for key, value in must_conditions.items()
            ]

        return Filter(should=should_filters, must=must_filters)
