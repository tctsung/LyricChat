import os
from dotenv import dotenv_values
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from datetime import datetime

# load helper functions:
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)    # append src/ to sys.path to load helper.py     
from helper import *

# get secret tokens:
env_path = os.path.join(os.path.dirname(parent_dir), ".env")  # load LyricChat/.env to env path
from dotenv import dotenv_values
ENV_VAR = dotenv_values(env_path)
HF_key = ENV_VAR['HF_key']
login(token=HF_key) 

def load_HF_classification_model(model_id = "SamLowe/roberta-base-go_emotions"):
    """
    TODO: save a HF classification model to local for emotion classification
    param model_id (str): HF model API id
    """
    # define save path:
    save_path="models/HF/" + clean_file_name(model_id)
    # get model and tokenizer from HF API:
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

class LoadGenius:
    input_directory="data/genius"
    # TODO: create a combined file for scraped genius.com data | lyric-database-master
    def __init__(self, file_name = "lyrics_processed.parquet"):
        """
        param input_directory (str): directory of the scraped lyrics
        """
        self.file_name = file_name
        self.list_latest_dirs()
        
        # iterate over the latest folders for feature engineering
    def list_latest_dirs(self):
        dirs = os.listdir(LoadGenius.input_directory)
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
        self.latest_dirs = {artist: os.path.join(LoadGenius.input_directory, latest_folder, self.file_name) for artist, (timestamp, latest_folder) in dct.items()}

class EmotionClassifier:
    # TODO: do emotion classification, no matter the source
    def __init__(self, parquet_path, model_path = 'models/HF/SamLowe_roberta_base_go_emotions', thread=4, save=False):
        """
        param parquet_path (str): path to the parquet file with unique id & lyrics
        param model_path (str): local path to the HF classification model, this is the output of load_HF_classification_model()
        return: a new parquet & csv file with extra 3 cols: primary_emotion, secondary_emotion, classification_result
        """
        self.thread = thread
        # load data:
        self.parquet_path = parquet_path
        self.load_data()
        # load model:
        self.model_path = model_path
        self.load_model()
        # do emotion classification:
        self.classification()
        if save:
            self.save()

    def load_data(self):
        self.df = pd.read_parquet(self.parquet_path)
        # check colnames has "id" & "lyrics":
        assert ("id" in self.df.columns) and ("lyrics" in self.df.columns), "Input data must have 'id' & 'lyrics' columns"
        self.df['lyrics_clean'] = self.df['lyrics'].apply(clean_lyrics)
        self.hf_dataset = Dataset.from_pandas(self.df[["id", "lyrics_clean"]])
    def load_model(self, top_k=6):
        batch_size = self.thread
        self.classifier = pipeline(model = self.model_path, task = "text-classification", 
                      device_map="auto", top_k=top_k, batch_size=batch_size)
    def classification(self):
        classify_lst = []
        classification_results = self.classifier(KeyDataset(self.hf_dataset, "lyrics_clean"), truncation='longest_first')
        # organize results:
        for classify_dct in classification_results:
            clean_result = top_k_without_neutral(classify_dct, k=3)
            classify_lst.append(clean_result)
        # add results to dataframe:
        df_classification = pd.DataFrame(classify_lst, columns=['primary_emotion', 'secondary_emotion', 'tertiary_emotion', 'classification_result'])
        self.df = pd.concat([self.df, df_classification], axis=1)
    def do_summary(self):
        self.primary_emotion = self.df["primary_emotion"].value_counts().to_dict()
        self.secondary_emotion = self.df["secondary_emotion"].value_counts().to_dict()
        self.tertiary_emotion = self.df["tertiary_emotion"].value_counts().to_dict()
        # combine two emotions into one:
        combined_series = pd.concat([self.df["primary_emotion"], self.df["secondary_emotion"], self.df["tertiary_emotion"]])
        self.total_emotion = combined_series.value_counts().to_dict()
    def save(self):
        directory = os.path.dirname(self.parquet_path)
        file_path = os.path.join(directory, 'emotion.parquet')
        self.df.to_parquet(file_path)
        file_path = os.path.join(directory, 'emotion.csv')
        self.df.to_csv(file_path)
## Helper functions:
def clean_lyrics(lyrics):
    # TODO: remove metadat & delimiter from lyric chunks
    output = lyrics.replace("<\Lyric>```", "")     # for last idx
    output = output.split("<Lyric>")[-1].strip()   # for first idx
    return output
def top_k_without_neutral(x, k=3):
    # TODO: get top k classification results without neutral emotion
    output = []
    for classify_dct in x:
        if classify_dct['label'] != 'neutral':
            output.append(classify_dct['label'])
        if len(output) == k:
            output.append(str(x))
            return output
    
