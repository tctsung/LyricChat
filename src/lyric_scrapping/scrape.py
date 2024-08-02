import lyricsgenius
from difflib import SequenceMatcher as sm
import pandas as pd
import numpy as np
import helper
import logging
import os
import contextlib
import json
import time

# get secret tokens:
from dotenv import dotenv_values
ENV_VAR = dotenv_values(".env")
Genius_key = ENV_VAR['Genius_key']

# Run this code in the git repo with: `python src/lyric_scrapping/scrape.py`
def main():
    # set parameters:
    artist_name = "Imagine Dragons"
    sort_method = "title"
    max_songs = 500
    logging_level = 'INFO'
    save_directory = "data\lyrics"

    # process scraping:
    helper.set_loggings(level=logging_level, func_name="LyricScraper")
    logging.info("Start scraping %d songs for %s", max_songs, artist_name)
    scraper = LyricScraper(artist_name, max_songs=max_songs, sort_method=sort_method)
    scraper.save(save_directory)

class LyricScraper:
    def __init__(self, artist_name, max_songs=500, sort_method="title", check_similar_songs=True, threshold=0.6):
        self.artist_name = artist_name
        self.max_songs = max_songs
        self.sort_method = sort_method
        # process scraping:
        self.start_time = helper.get_timestamp()
        self.scrape_songs()
        if check_similar_songs:
            self.rm_similar_songs(threshold=threshold)
        
    def scrape_songs(self):
        # collect required args:
        genius = lyricsgenius.Genius(Genius_key)   # global variable
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            artist_id = genius.search_artist(self.artist_name, max_songs=0, include_features=False).id 
    
        # scrape songs:
        while True:
            try:
                artist = genius.search_artist(self.artist_name, artist_id=artist_id, max_songs=self.max_songs, sort=self.sort_method)
                break
            except:
                logging.warning("Failed to scrape songs. Retrying...")
                time.sleep(3)
                continue
        # save info:
        self.artist = artist
        self.raw_dict = {song.title: song.lyrics for song in artist.songs}

    def check_similarity(self, threshold=0.6, include_raw=False):
        # calculate the similarity:
        if include_raw:
            lyrics = [song.lyrics for song in self.artist.songs]
            titles = [song.title for song in self.artist.songs]
        else:
            lyrics = [song.lyrics for song in self.artist.songs if is_valid_song(song.title)]
            titles = [song.title for song in self.artist.songs if is_valid_song(song.title)]
            logging.warning("%d songs were removed due to invalid title", 
                        len(self.raw_dict) - len(titles))
        n = len(titles)
        similarity_matrix = np.ones((n, n))  
        for i in range(n):
            for j in range(i+1, n):  # We only need to fill the upper triangle
                similarity = max(
                    sm(None, lyrics[i], lyrics[j]).ratio(),
                    sm(None, lyrics[j], lyrics[i]).ratio()
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        similarity_df = pd.DataFrame(similarity_matrix, index=titles, columns=titles)

        # identify the ones with high similarity:
        similar_songs = []
        for index, row in similarity_df.iterrows():
            above_threshold = row.index[row>threshold].to_list()
            if len(above_threshold) > 1:
                similar_songs.append(above_threshold)
        removed_similar_songs = []
        for songs in similar_songs:
            songs.sort(key=len)
            removed_similar_songs.extend(songs[1:])
        removed_similar_songs = set(removed_similar_songs)
        logging.warning("%d songs were removed due to high similarity, check LyricScraper.removed_songs for more info", 
                        len(removed_similar_songs))
        self.similar_songs = similar_songs
        self.similarity_df = similarity_df
        self.removed_similar_songs = removed_similar_songs
    def rm_similar_songs(self, threshold=0.6, check_all=False):
        self.check_similarity(threshold=threshold, include_raw=check_all)
        filter_dict = {}
        removed_songs = []
        for title, lyric in self.raw_dict.items():
            if (title not in self.removed_similar_songs) and is_valid_song(title):
                filter_dict[title] = lyric
            else:
                removed_songs.append(title)
        self.removed_songs = removed_songs   # include invalid songs in removed_songs
        self.filter_dict = filter_dict
    def create_metadata(self):
        self.end_time = helper.get_timestamp()
        return f"""Start time: {self.start_time}
End time: {self.end_time}
Artist name: {self.artist_name}
Saved to: {self.save_directory}
Total # of Songs scraped: {len(self.raw_dict)}, see lyrics_raw.json for more info
Total # of filtered Songs: {len(self.filter_dict)}, see lyrics_filter.json for more info
Removed due to simialrity check: {len(self.removed_similar_songs)}
Removed due to invalid title: {len(self.removed_songs)-len(self.removed_similar_songs)}
"""
    def save(self, directory):
        subfolder = helper.clean_file_name(f"{self.artist_name} {self.start_time}") 
        self.save_directory = os.path.join(directory, subfolder)
        os.makedirs(self.save_directory, exist_ok=True)
        
        # save raw_dict to json:
        with open(os.path.join(self.save_directory, 'lyrics_raw.json'), 'w') as fp:
            json.dump(self.raw_dict, fp)
        # save filter_dict to json:
        with open(os.path.join(self.save_directory, 'lyrics_filter.json'), 'w') as fp:
            json.dump(self.filter_dict, fp)
        # save metadata to LyricScraper.meta:
        with open(os.path.join(self.save_directory, 'LyricScraper.meta'), 'w') as fp:
            fp.write(self.create_metadata())

        logging.info("Files saved to %s, see LyricScraper.meta for more info", self.save_directory)

#### helper functions ####
def is_valid_song(title):
    excluded_terms = ["remix)", "(demo)", "(live", "session)", "version)"]  
    return not any(term.lower() in title.lower() for term in excluded_terms)
#### End of helper functions ####

if __name__ == '__main__':
	main()      # the func to run