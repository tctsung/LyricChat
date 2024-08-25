import guidance
from guidance import models, gen, select, silent
from guidance import system, user, assistant
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm          # progress bar
import pandas as pd
import numpy as np

# Potential future work
# 1. Use LLM to decide the emotion categories based on Artist
# 2. Let lamma-cpp-python use GPU for faster inference
# 3. Do parallel processing or batch processing in inference

def main():       # python .\src\emotion_feature_extraction\extract.py
    # set args:
    lyric_path = "data\lyrics\THE_SCRIPT_2024_08_13_16_25_33\lyrics_filter.json"
    gguf_path = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
    logging_level = 'INFO'

    # dp feature extraction:
    set_loggings(level=logging_level, func_name="EmotionExtractor")
    meta_extractor = EmotionExtractor(lyric_path, gguf_path)

class EmotionExtractor:
    """
    TODO: classify emotions of given texts
    """
    # emotion for lyric classification recommended by Claude (automate this process based on singers in the future)
    emotions = ['Joy/Happiness', 'Love/Affection', 'Nostalgia/Melancholy', 'Sadness/Sorrow',
                'Anger/Frustration', 'Fear/Anxiety', 'Hope/Optimism', 'Longing/Desire',
                'Empowerment/Confidence','Disappointment/Regret', 'Contentment/Peace', 'Excitement/Enthusiasm',
                'Loneliness/Isolation', 'Gratitude/Appreciation', 'Confusion/Uncertainty']
    sentiments = ['Positive', 'Negative', 'Neutral']
    def __init__(self, lyric_path, gguf_path):
        # arg check:
        assert lyric_path.endswith('.json'), "lyric_path must end with .json"
        
        # collect required args:
        self.start_time = get_timestamp()
        self.lyric_path = lyric_path
        self.gguf_path = gguf_path
        with open(lyric_path, 'r') as fp:
            self.lyrics = json.load(fp)
        self.llm = models.LlamaCpp(model=gguf_path, n_ctx=4096, echo=False) 
        self.lyric_feature = {}  # buffers
        self.artist = get_artist_name(lyric_path)
        # do extraction:
        self.drop_duplicates()          # rm duplicated lyrics
        self.extract_lyric_feature()
        self.do_summary()               # summary stats of features
        # save data:
        self.save()
    def drop_duplicates(self):
        """
        TODO: drop the exact same lyric chunks
        """
        for title, lyric in self.lyrics.copy().items():
            seen = set()
            lyric_chunk = lyric.split("\n\n")
            lyric_no_dup = "\n\n".join([x for x in lyric_chunk if not (x in seen or seen.add(x))])
            self.lyrics[title] = lyric_no_dup
    def extract_lyric_feature(self):
        """
        TODO: Use LLM to extract lyric features
        Future work: turn loop into batch processing/parallel processing
        """
        idx = 0
        for title, lyric in tqdm(self.lyrics.items()):
            idx += 1
            if idx % 10 == 0:
                logging.info("Current progress: %d/%d songs", idx, len(self.lyrics))
            with silent():
                # add system prompt:
                with system():
                    lm = self.llm + """You are an Sentiment Analysis expert and lyric metadata extractor. 
    Analyze the following lyrics and extract information, including sentiments, emotional tone, and key themes.\n\n\n
    """
                # get model response:
            lm += llm_extraction(self.artist, title, lyric)   
            self.lyric_feature[title] = {
                "sentiment": lm["sentiment"],
                "primary_emotion": lm["primary_emotion"],
                "secondary_emotion": lm["secondary_emotion"],
                "theme": lm["theme"]
            }
    def do_summary(self):
         # Get summary statistics of extracter features
         feature_df = pd.DataFrame(self.lyric_feature).T   # turn json into df
         self.sentiment = feature_df["sentiment"].value_counts().to_dict()
         self.primary_emotion = feature_df["primary_emotion"].value_counts().to_dict()
         self.secondary_emotion = feature_df["secondary_emotion"].value_counts().to_dict()
         # combine two emotions into one:
         combined_series = pd.concat([feature_df["primary_emotion"], feature_df["secondary_emotion"]])
         self.total_emotion = combined_series.value_counts().to_dict()
    def create_metadata(self):
        # TODO: create metadata for the feature extraction process
        self.end_time = get_timestamp()
        return f"""Start time: {self.start_time}
End time: {self.end_time}
Features explanation:
- sentiment categories: {EmotionExtractor.sentiments}
- emotion categories: {EmotionExtractor.emotions}
- theme: A string of 3 keywords representing the theme of the lyric; seperated by comma & "and"
Summary:
Total songs: {len(self.lyrics)}
- sentiment: {self.sentiment}
- total_emotion: {self.total_emotion}
- primary_emotion: {self.primary_emotion}
- secondary_emotion: {self.secondary_emotion}
File location:
Model: {self.gguf_path}
Input & output data directory: {self.directory} 
Input file: {os.path.basename(self.lyric_path)}
Output file: lyrics_feature.json
"""
        
    def save(self):
        # save files to local
        self.directory = os.path.dirname(self.lyric_path)
        with open(os.path.join(self.directory, 'lyrics_feature.json'), 'w') as fp:
            json.dump(self.lyric_feature, fp, indent=4)
        with open(os.path.join(self.directory, 'EmotionExtractor.meta'), 'w') as fp:
            fp.write(self.create_metadata())
        logging.info("File `lyrics_feature.json` & `EmotionExtractor.meta` saved to: %s", self.directory)

#### Helper functions ####
@guidance
def llm_extraction(lm, artist, title, lyric): 
    # self-defined func for prompt engineering:
    # TODO: use LLM to extract lyric features
    pattern = "[a-zA-Z -]+, [a-zA-Z -]+, and [a-zA-Z -]+"   # pattern for 3 keywords: eg. Self-Discovery, Heartbreak, Letting Go
    with silent():
        lm += f"""
* Provided song information:
Artist: {artist}
Title: {title}
Lyric: 
```
{lyric}
```
\n
* Sentiment Analysis

Overall Sentiment: {select(EmotionExtractor.sentiments, name = 'sentiment')}
Primary Emotion: {select(EmotionExtractor.emotions, name='primary_emotion')}
Secondary Emotion (this must be different to the primary emotion): {select(EmotionExtractor.emotions, name='secondary_emotion')}
Theme/keyword: use three keywords to describe the theme of the above lyric: {gen(name="theme", regex=pattern, temperature=0.2)}"""
    return lm

def set_loggings(level=logging.INFO, func_name=''):
	"""
	TODO: set logging levels
	"""
	if isinstance(level, str):
		log_levels = {
			'DEBUG': logging.DEBUG,
			'INFO': logging.INFO,
			'WARNING': logging.WARNING,
			'ERROR': logging.ERROR,
			'CRITICAL': logging.CRITICAL
		}
		level = log_levels[level]
	# Remove all handlers associated with the root logger object:
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(
		level=level,                              # set logging level
		format='----- %(levelname)s (%(asctime)s) ----- \n%(message)s\n')	 # set messsage format
	logging.critical(
		'Hello %s, The current logging level is: %s', 
		func_name,
		logging.getLevelName(logging.getLogger().getEffectiveLevel()))

def get_timestamp():
	current_timestamp = datetime.now()
	return current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
def get_artist_name(path):
    # TODO: get artist name from LyricScraper.meta
    directory = os.path.dirname(path)
    with open(os.path.join(directory, 'LyricScraper.meta'), 'r') as fp:
        for line in fp:
            if line.startswith('Artist name: '):
                return line.split(':')[1].strip()
##### End of helper functions ####

if __name__ == '__main__':
	main()      # the func to run