import os
from dotenv import dotenv_values
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from datasets import Dataset
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

def load_HF_classification_model(model_id = "michellejieli/emotion_text_classifier"):
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
    def __init__(self, thread=2):
        """
        param input_directory (str): directory of the scraped lyrics
        """
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
        self.latest_genius_dirs = {artist: os.path.join(LoadGenius.input_directory, latest_folder, "lyrics_processed.json") for artist, (timestamp, latest_folder) in dct.items()}

class EmotionClassifier:
    # TODO: do emotion classification, no matter the source
    def __init__(self, json_path, artist, model_path = 'models/HF/michellejieli_emotion_text_classifier', type="genius"):
        """
        param json_path (str): path to the jsob file with format- <song title>: <lyrics>
        param artist (str): name of the artist, this is provided because format of genius.com & lyric-database-master are different
        return: lyrics_feature.json with format- <original key>: [emotion1, emotion2, emotion3, raw data]
        """
        self.json_path = json_path
        self.artist = artist
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

    def load_dataset(self):
        with open(os.path.join(dir, 'lyrics_filter.json'), 'r') as fp:
            lyrics = json.load(fp)
        dataset = Dataset.from_dict({"text": data})


## Helper functions:
def clean_lyrics(lyrics):
    # TODO: remove metadat & delimiter from lyric chunks
    output = lyrics.replace("<\Lyric>```", "")     # for last idx
    output = output.split("<Lyric>")[-1].strip()   # for first idx
    return output

