import os

from dotenv import dotenv_values
from huggingface_hub import login
from transformers import RobertaForSequenceClassification
import json
from datasets import Dataset
from datetime import datetime

# load HF key:
ENV_VAR = dotenv_values(".env")
HF_key = ENV_VAR['HF_key']
login(token=HF_key) 

# load helper:
import sys
sys.path.append("src/")      # add path
import helper

# model = RobertaForSequenceClassification.from_pretrained("models/HF/roberta-base-go_emotions")


class LoadGenius:
    input_directory="data/genius"
    # TODO: create a combined file for scraped genius.com data | lyric-database-master
    def __init__(self):
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
        self.latest_genius_dirs = {artist: os.path.join(LoadGenius.input_directory, latest_folder, "lyrics_filter.json") for artist, (timestamp, latest_folder) in dct.items()}

class EmotionClassifier:
    # TODO: do emotion classification, no matter the source
    def __init__(self, json_path, artist, model_path = "models/HF/roberta-base-go_emotions", type="genius"):
        """
        param json_path (str): path to the jsob file with format- <song title>: <lyrics>
        param artist (str): name of the artist, this is provided because format of genius.com & lyric-database-master are different
        return: lyrics_feature.json with format- <artist>_<song title>: []
        """
        self.json_path = json_path
        self.artist = artist
        self.model_path = model_path
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)

    def load_dataset(self):
        with open(os.path.join(dir, 'lyrics_filter.json'), 'r') as fp:
            lyrics = json.load(fp)
        dataset = Dataset.from_dict({"text": data})


