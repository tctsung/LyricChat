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
    if len(sys.argv) != 4:
        raise ValueError("Usage: python scrape.py <scrape_type> <artist_name> <max_songs>")

    scrape_type, artist_name, max_songs = sys.argv[1:4]
    assert scrape_type in ("genius", "open-lyrics"), "Scrape type must be either 'genius' or 'open-lyrics'"
    assert (max_songs.isdigit()) and (int(max_songs)>0), "Maximum number of songs must be an integer"

    # do scraping:
    if scrape_type == "genius":
        Genius_key = ENV_VAR['Genius_key']   # load Genius API key from .env file
        set_loggings(level="info", func_name="Genius song Scraper")
        genius_scraper = GeniusScraper(artist_name, max_songs=int(max_songs), Genius_key=Genius_key)
        lyrics_raw = genius_scraper.raw_dict
    else:  # scrape_type == "open-lyrics":
        set_loggings(level="info", func_name="Open-Lyrics Scraper")

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
            self.save(directory = "data\lyrics")
        logging.info("Total time taken: %s\nTotal songs scraped: %s\nFile saved to: %s", 
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
        # save info:
        self.artist = artist
        self.raw_dict = {song.title: song.lyrics for song in artist.songs}
    
    def save(self, directory):
        # TODO: save lyrics as json
        subfolder = clean_file_name(f"{self.artist_name} {get_timestamp()}")
        self.save_directory = os.path.join(directory, subfolder)
        os.makedirs(self.save_directory, exist_ok=True)
        self.save_path = os.path.join(self.save_directory, 'lyrics_raw.json')
        with open(self.save_path, 'w') as fp:
            json.dump(self.raw_dict, fp, indent=4)


class LyricProcessor:
    # commonly seen irrelevant words in Genius lyrics:
    default_irrelevant_words = [
        r"You might also like",
        r"\d*Embed$"
    ]
    def __init__(self, artist_name, max_songs, sort_method="title", irrelevant_words=[], filter_songs=True, similarity_threshold=0.6,
                 retry_times=10, save=True):
        """
        TODO: scrape lyrics of a specific artist from Genius API
        param artist_name (str): name of the artist
        param max_songs (int): maximum number of songs to scrape
        param sort_method (str): sort method of the songs
        param check_similar_songs (bool): whether to remove songs with similar lyrics, the shortest song name will be kept
        param threshold (float): threshold of similarity
        """
        # set params:
        self.artist_name = artist_name
        self.max_songs = max_songs
        self.sort_method = sort_method
        self.irrelevant_words = LyricScraper.default_irrelevant_words
        self.retry_times = retry_times
        if len(irrelevant_words) > 0:
            self.irrelevant_words.extend(irrelevant_words)

        # set buffers:
        self.removed_songs = []
        self.invalid_songs = []
        self.highly_similar_songs = []
        self.inadquate_len_songs = []

        # process scraping:
        self.start_time = get_timestamp()
        self.scrape_songs()
        self.preprocess()
        if filter_songs:
            self.removed_invalid_titles()
            self.removed_inadequate_word_cnt()
            self.removed_similar_songs(threshold=similarity_threshold)
        
        # save files:
        if save: 
            self.save(directory = "data\lyrics")
        
    def scrape_songs(self):
        # TODO: scrape lyrics from Genius API
        # collect required args:
        genius = lyricsgenius.Genius(Genius_key)   # global variable
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            artist_id = genius.search_artist(self.artist_name, max_songs=0, include_features=False).id 
    
        # scrape songs:
        while True:       # if failed due to internal error, retry
            try:
                artist = genius.search_artist(self.artist_name, artist_id=artist_id, max_songs=self.max_songs, sort=self.sort_method)
                break
            except:
                if self.retry_times == 0:    # stop if no more retries
                    logging.error("Failed to scrape songs. Maximum retry times reached.")
                    exit()
                logging.warning("Failed to scrape songs. Retrying...")
                self.retry_times -= 1
                time.sleep(3)
                continue
        # save info:
        self.artist = artist
        self.raw_dict = {song.title: song.lyrics for song in artist.songs}
    def preprocess(self):
        # TODO: preprocess the lyrics
        processed_dict = {}
        for title, lyric in self.raw_dict.items():
            # split by "Lyrics", and remove the first part (Contributors & song name)
            lyric_process = lyric.split("Lyrics")[1:]
            lyric_process = "".join(lyric_process)

            # remove [XXX] such as [Chorus], [Verse 1] {Verse 1} {Bridge]
            lyric_process = re.sub(r'(\{|\[).*?(\]|\})', '', lyric_process)

            # remove irrlevant patterns:
            irrelevant_merge = '|'.join(self.irrelevant_words)
            irrelevant_regex = re.compile(irrelevant_merge)
            lyric_process = re.sub(irrelevant_regex, '', lyric_process)

            # turn multiple new lines into single new line to split chunks:
            lyric_process = re.sub(r'\n\n\n+', '\n\n', lyric_process)

            # remove leading and trailing spaces:
            lyric_process = lyric_process.strip()
            
            # save processed lyrics:
            processed_dict[title] = lyric_process
        self.processed_dict = processed_dict
    def removed_invalid_titles(self):
        # TODO: remove invalid titles from processed_dict
        for title in self.processed_dict.copy().keys():
            if not is_valid_song(title):
                self.processed_dict.pop(title)
                self.invalid_songs.append(title)
                self.removed_songs.append(title)
        if len(self.invalid_songs) > 0:
            logging.warning("%d songs were removed due to invalid title", len(self.invalid_songs))
    def removed_inadequate_word_cnt(self, min_word_cnt=30):
        # TODO: remove songs with inadequate word count since that might not be lyrics
        for title, lyric in self.processed_dict.copy().items():
            if len(lyric.split()) < min_word_cnt:
                self.processed_dict.pop(title)
                self.inadquate_len_songs.append(title)
                self.removed_songs.append(title)
        if len(self.inadquate_len_songs) > 0:
            logging.warning("%d songs were removed due to inadequate word count", len(self.inadquate_len_songs))

    def check_similarity(self, threshold=0.6, do_raw=False):
        # TODO: calculate the similarity based on longest contiguous matching subsequence (LCS) algorithm
        # This is a helper function for removed_similar_songs()
        if do_raw:  # use all songs to calculate similarity
            lyric_dict = self.raw_dict
        else:
            lyric_dict = self.processed_dict
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
        for title in self.processed_dict.copy().keys():
            if title in highly_similar_songs:
                self.removed_songs.append(title)
                self.highly_similar_songs.append(title)
                self.processed_dict.pop(title)
        if len(self.highly_similar_songs) > 0:
            logging.warning("%d songs were removed due to similar lyrics", len(self.highly_similar_songs))
    def drop_duplicates(self):
        """
        TODO: helper: drop the exact same lyric chunks
        """
        for title, lyric in self.processed_dict.copy().items():
            seen = set()
            lyric_chunk = lyric.split("\n\n")
            lyric_no_dup = "\n\n".join([x for x in lyric_chunk if not (x in seen or seen.add(x))])
            self.processed_dict[title] = lyric_no_dup
    def create_metadata(self):
        # TODO: create metadata for the scraped lyrics
        self.end_time = get_timestamp()
        return f"""Start time: {self.start_time}
End time: {self.end_time}
Artist name: {self.artist_name}
Saved to: {self.save_directory}
OOP object: LyricScraper.pickle
Total # of Songs scraped: {len(self.raw_dict)} (see lyrics_raw.json for more info)
Total # of filtered Songs: {len(self.processed_dict)} (see lyrics_filter.json for more info)
Removed due to invalid title: {len(self.invalid_songs)} (see LyricScraper.invalid_songs for more info)
Removed due to inadequate word count: {len(self.inadquate_len_songs)} (see LyricScraper.inadquate_len_songs for more info)
Removed due to simialrity check: {len(self.highly_similar_songs)} (see LyricScraper.highly_similar_songs for more info)
"""
    def save(self, directory = "data\lyrics"):
        # TODO: save the lyrics, OOP object, and metadata to specified directory
        subfolder = clean_file_name(f"{self.artist_name} {self.start_time}") 
        self.save_directory = os.path.join(directory, subfolder)
        os.makedirs(self.save_directory, exist_ok=True)
        
        # save raw_dict to json:
        with open(os.path.join(self.save_directory, 'lyrics_raw.json'), 'w') as fp:
            json.dump(self.raw_dict, fp, indent=4)
        # save processed_dict to json:
        with open(os.path.join(self.save_directory, 'lyrics_filter.json'), 'w') as fp:
            json.dump(self.processed_dict, fp, indent=4)
        # save metadata to LyricScraper.meta:
        with open(os.path.join(self.save_directory, 'LyricScraper.meta'), 'w') as fp:
            fp.write(self.create_metadata())
        # save the OOP itself as pickle:
        with open(os.path.join(self.save_directory, 'LyricScraper.pickle'), 'wb') as f:
            pickle.dump(self, f)

        logging.info("Files saved to %s, see LyricScraper.meta for more info", self.save_directory)

#### helper functions ####
def is_valid_song(title):
    # TODO: check if the title is valid
    excluded_terms = ["(acoustic", "[acoustic", "remix)", "remix]", "(demo", "[demo", "[live", "(live", "session)", "session]", "version)", "version]"]  
    return not any(term.lower() in title.lower() for term in excluded_terms)
#### End of helper functions ####

if __name__ == '__main__':
	main()      # the func to run