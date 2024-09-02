import lyricsgenius
from difflib import SequenceMatcher as sm
import pandas as pd
import numpy as np
import sys
import logging
import os
import contextlib
import json
import time
import re
import pickle
from datetime import datetime

# set working directory to LyricChat repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir('../..')     

# load helper functions:
import sys
sys.path.append("src/")     
from helper import *

# get secret tokens:
from dotenv import dotenv_values
ENV_VAR = dotenv_values(".env")


# Run this code in the git repo with: `python src/lyric_scrapping/scrape.py`
# TODO: separate lyrics scrapping & preprocessing into 2 OOP
# lyrics_raw: must be exact raw data (better for troubleshooting);  <artist>_<song title>: <lyrics>
# add a OOP for github lyric-database

# syntax: python scrape.py <scrape_type> <artist> <max_song>
# eg. python src/lyric_scrapping/scrape.py genius "Imagine Dragons" 10
def main():        
    # collect system args:
    set_loggings(level="info", func_name="Hello Lyric Scraper!")

class OpenLyricsScraper:
    def __init__(self):
        pass
        # format of key: <artist>|||<song title>|||<scraped order>

class GeniusScraper:
    def __init__(self, artist_name, max_songs, Genius_key, retry_times=10, save=True):
        """
        TODO: scrape lyrics of a specific artist from Genius API
        param artist_name (str): name of the artist
        param max_songs (int): maximum number of songs to scrape
        param sort_method (str): sort method of the songs
        """
        # set params:
        self.artist_name = artist_name
        self.max_songs = max_songs
        self.Genius_key = Genius_key
        self.retry_times = retry_times
        self.start_time = datetime.now()
        self.scrape_songs()   # scrape songs
        if save: 
            self.save()
        logging.info("Total time taken: %s\nTotal songs scraped: %s\nScraped lyrics saved to: %s", 
                     format_timedelta(datetime.now() - self.start_time), 
                     len(self.raw_dict),
                     self.save_path
                     )
    def scrape_songs(self):
        # TODO: scrape lyrics from Genius API
        # collect required args:
        genius = lyricsgenius.Genius(self.Genius_key)   # global variable
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # suppress stdout
            artist_id = genius.search_artist(self.artist_name, max_songs=0, include_features=False).id 
    
        # scrape songs:
        while True:       # if failed due to internal error, retry
            try:
                artist = genius.search_artist(self.artist_name, artist_id=artist_id, max_songs=self.max_songs, sort="title")
                break
            except:
                if self.retry_times == 0:    # stop if no more retries
                    logging.error("Failed to scrape songs. Maximum retry times reached.")
                    exit()
                logging.warning("Failed to scrape songs. Retrying...")
                self.retry_times -= 1
                time.sleep(3)
                continue
        # update artist name by more standardized Genius renaming
        self.artist_name = artist.name
        # save info:
        self.artist = artist
        # unique KEY: <artist>_<song title>_<scraped order>: <lyrics>
        self.raw_dict = {(self.artist_name+'|||'+song.title + '|||' + str(i)): song.lyrics for i, song in enumerate(artist.songs)}
    
    def save(self, directory = "data/genius"):
        # TODO: save lyrics as json
        subfolder = clean_file_name(f"{self.artist_name} {get_timestamp()}")
        self.save_directory = os.path.join(directory, subfolder)
        os.makedirs(self.save_directory, exist_ok=True)
        self.save_path = os.path.join(self.save_directory, 'lyrics_raw.json')
        with open(self.save_path, 'w', encoding='utf-8') as fp:
            json.dump(self.raw_dict, fp, indent=4)

class LyricProcessor:
    # commonly seen irrelevant words/commercials in Genius.com:
    default_irrelevant_words = [
        r"You might also like",
        r"\d*Embed$",
        r"See .+ Live",
        r"Get tickets as low as \$\d+"
    ]
    def __init__(self, json_path, scrape_type, check_invalid_title=True, check_word_cnt=True, check_lyric_similarity=True, similarity_threshold=0.6, irrelevant_words=[]):
        """
        TODO: scrape lyrics of a specific artist from Genius API
        param json_path (str | dict): json output of GeniusScraper or OpenLyricsScraper
        param max_songs (int): maximum number of songs to scrape
        param sort_method (str): sort method of the songs
        param check_similar_songs (bool): whether to remove songs with similar lyrics, the shortest song name will be kept
        param threshold (float): threshold of similarity
        """
        assert scrape_type in ["genius", "open-lyrics"], "scrape_type must be 'genius' or 'open-lyrics'"
        # set params:
        self.irrelevant_words = LyricProcessor.default_irrelevant_words
        if len(irrelevant_words) > 0:
            self.irrelevant_words.extend(irrelevant_words)
        # read input data:
        if isinstance(json_path, dict):
            self.lyrics_raw = json_path
        elif isinstance(json_path, str):
            assert json_path.endswith(".json"), "json_path must be a .json file."
            with open(json_path, 'r') as f:
                self.lyrics_raw = json.load(f)
        else:
            raise TypeError("json_path must be a .json file or a dict.")
        # set buffers:
        self.removed_songs = []
        self.invalid_songs = []
        self.highly_similar_songs = []
        self.inadquate_len_songs = []
        self.applied_filters = []
        # process scraping:
        self.start_time = get_timestamp()
        if scrape_type == "genius":
            self.preprocess_genius()
        elif scrape_type == "open-lyrics":
            self.preprocess_open_lyrics()

        # do filtering:
        if check_invalid_title:
            self.applied_filters.append("removed_invalid_titles")
            self.removed_invalid_titles()
        if check_word_cnt:
            self.applied_filters.append("removed_inadequate_word_cnt")
            self.removed_inadequate_word_cnt()
        if check_lyric_similarity:
            self.applied_filters.append("removed_highly_similar_songs with threshold" + str(similarity_threshold))
            self.removed_similar_songs(threshold=similarity_threshold)
        
        # split to child chunks (for small2big retrieval):
        self.split_to_chunks()
        # create output files:
        self.create_metadata()
    
    def split_to_chunks(self):
        # TODO: split lyrics into child chunks
        lyrics_chunks = {}
        for key, lyrics in self.lyrics_processed.items():
            key_split = key.split("|||")     # <artist>|||<song title>|||<scraped order>
            artist_name, song_title = key_split[0], key_split[1]   
            chunks = lyrics.split('\n\n')   # split parent lyrics to child chunks
            for idx, chunk in enumerate(chunks):
                new_key = f"{key}|||{idx}"   # consistent delimiter
                chunk_data = {
                "artist_name": artist_name,
                "song_title": song_title,
                "lyrics": chunk,
                "parent_id": key,
                "chunk_order": idx
                }
                if idx == 0:                 # add artist name & song title to the first chunk for LLM to read
                    chunk_data["lyrics"] = f"```\n<Artist name> {artist_name} <\Artist>\n<Song title> {song_title} <\Song title>\n<Lyric> \n{chunk}"
                elif idx == len(chunks)-1:   # add delimiter to the last chunk
                    chunk_data["lyrics"] = chunk_data['lyrics'] + "\n<\Lyric>```"
                lyrics_chunks[new_key] = chunk_data    # save to new dict
        self.lyrics_chunks = lyrics_chunks

    def preprocess_open_lyrics(self):
        # add this after implementing open-lyrics scraper
        # note: remember to create unique key 
        
        pass
    def preprocess_genius(self):
        # TODO: preprocess the lyrics
        lyrics_processed = {}
        for title, lyric in self.lyrics_raw.items():
            # split by "Lyrics", and remove the first part (Contributors & song name)
            lyric_process = lyric.split("Lyrics")[1:]
            lyric_process = "".join(lyric_process)

            # remove [XXX] such as [Chorus], [Verse 1] {Verse 1} {Bridge]
            lyric_process = re.sub(r'(\{|\[).*?(\]|\})', '', lyric_process)

            # remove irrlevant patterns:
            irrelevant_merge = '|'.join(self.irrelevant_words)
            irrelevant_regex = re.compile(irrelevant_merge)
            lyric_process = re.sub(irrelevant_regex, '', lyric_process)

            # turn multiple new lines into double new line to split chunks:
            lyric_process = re.sub(r'\n\n\n+', '\n\n', lyric_process)

            # remove leading and trailing spaces:
            lyric_process = lyric_process.strip()

            # remove duplicated chunks:
            lyric_process = drop_duplicates(lyric_process)

            # save processed lyrics:
            lyrics_processed[title] = lyric_process
        self.lyrics_processed = lyrics_processed
    def removed_invalid_titles(self):
        # TODO: remove invalid titles from lyrics_processed
        for title in self.lyrics_processed.copy().keys():
            if not is_valid_song(title):
                self.lyrics_processed.pop(title)
                self.invalid_songs.append(title)
                self.removed_songs.append(title)
        if len(self.invalid_songs) > 0:
            logging.warning("%d songs were removed due to invalid title", len(self.invalid_songs))
    def removed_inadequate_word_cnt(self, min_word_cnt=30):
        # TODO: remove songs with inadequate word count since that might not be lyrics
        for title, lyric in self.lyrics_processed.copy().items():
            if len(lyric.split()) < min_word_cnt:
                self.lyrics_processed.pop(title)
                self.inadquate_len_songs.append(title)
                self.removed_songs.append(title)
        if len(self.inadquate_len_songs) > 0:
            logging.warning("%d songs were removed due to inadequate word count", len(self.inadquate_len_songs))
    def check_similarity(self, threshold=0.6, do_raw=False):
        # TODO: calculate the similarity based on longest contiguous matching subsequence (LCS) algorithm
        # This is a helper function for removed_similar_songs()
        lyric_dict = self.lyrics_processed
        titles = []
        lyrics = []
        for title, lyric in lyric_dict.items():
            titles.append(title)
            lyrics.append(lyric)
        n = len(titles)
        similarity_matrix = np.ones((n, n))  # buffer=1 since it's for similarity of song i & song i
        for i in range(n):
            for j in range(i+1, n):  # We only need to fill the upper triangle
                # calculate similarity based on LCS algorithm
                similarity = max(     
                    sm(None, lyrics[i], lyrics[j]).ratio(),
                    sm(None, lyrics[j], lyrics[i]).ratio()
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        similarity_df = pd.DataFrame(similarity_matrix, index=titles, columns=titles)

        # create a nested list for the ones with high similarity:
        similar_songs = []
        for index, row in similarity_df.iterrows():   # iterate rowswise
            above_threshold = row.index[row>threshold].to_list()  # keep the songs above threshold
            if len(above_threshold) > 1:
                similar_songs.append(above_threshold)   
        self.similar_songs = similar_songs            # a nested list of similar songs
        self.similarity_df = similarity_df
    def removed_similar_songs(self, threshold=0.6):
        # TODO: remove similar songs
        self.check_similarity(threshold=threshold, do_raw=False)  # get similarity matrix
        
        # identify highly similar songs:
        highly_similar_songs = []
        for songs in self.similar_songs:
            songs.sort(key=len)       # sort by length because shortest one is usually the original version
            highly_similar_songs.extend(songs[1:])      # eg. ["Believer", "Believer (Live)"] -> keep "Believer", drop live version
        highly_similar_songs = set(highly_similar_songs)
        
        # remove similar songs:
        for title in self.lyrics_processed.copy().keys():
            if title in highly_similar_songs:
                self.removed_songs.append(title)
                self.highly_similar_songs.append(title)
                self.lyrics_processed.pop(title)
        if len(self.highly_similar_songs) > 0:
            logging.warning("%d songs were removed due to similar lyrics", len(self.highly_similar_songs))
    def create_metadata(self):
        # TODO: create metadata for the scraped lyrics
        self.end_time = get_timestamp()
        self.metadata = f"""Start time: {self.start_time}
End time: {self.end_time}
Total # of input songs: {len(self.lyrics_raw)}
Total # of Processed Songs: {len(self.lyrics_processed)} (see lyrics_processed.json for more info)
Total # of song chunks: {len(self.lyrics_chunks)} (see lyrics_processed.json for more info)
Applyed filters: {self.applied_filters}
Removed due to invalid title: {len(self.invalid_songs)} (see LyricProcessor.invalid_songs for more info)
Removed due to inadequate word count: {len(self.inadquate_len_songs)} (see LyricProcessor.inadquate_len_songs for more info)
Removed due to simialrity check: {len(self.highly_similar_songs)} (see LyricProcessor.highly_similar_songs for more info)
"""
    def save(self, directory):
        # TODO: save the lyrics, OOP object, and metadata to specified directory
        # param directory (str): the directory to save files to, recommend use the same directory as GeniusScraper/OpenLyricsScraper
        
        # create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # save lyrics_processed to json:
        with open(os.path.join(directory, 'lyrics_processed.json'), 'w', encoding='utf-8') as fp:
            json.dump(self.lyrics_chunks, fp, indent=4)
        # save metadata to LyricProcessor.meta:
        with open(os.path.join(directory, 'LyricProcessor.meta'), 'w') as fp:
            fp.write(self.metadata)
        # save the OOP itself as pickle:
        with open(os.path.join(directory, 'LyricProcessor.pickle'), 'wb') as f:
            pickle.dump(self, f)

        logging.info("Files saved to %s, see LyricProcessor.meta for more info", directory)

#### helper functions ####
def is_valid_song(title):
    # TODO: check if the title is valid
    excluded_terms = ["(acoustic", "[acoustic", "remix)", "remix]", "(demo", "[demo", "[live", "(live", "session)", "session]", "version)", "version]"]  
    return not any(term.lower() in title.lower() for term in excluded_terms)
def drop_duplicates(lyric):
    """
    TODO: drop the exact same lyric chunks
    """
    seen = set()
    lyric_chunk = lyric.split("\n\n")
    lyric_no_dup = "\n\n".join([x for x in lyric_chunk if not (x in seen or seen.add(x))])
    return lyric_no_dup
#### End of helper functions ####

if __name__ == '__main__':
	main()      # the func to run