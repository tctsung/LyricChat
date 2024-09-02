import os
import sys

# set working directory to LyricChat repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir('../..')   

# load self-written functions:
sys.path.append("src/")     
from helper import *
sys.path.append("src/scrape/")
from scrape import *  

# get secret tokens:
from dotenv import dotenv_values
ENV_VAR = dotenv_values(".env")

def main():                   # eg. python runner.py genius "Lupe Fiasco" 10
    # collect system args:
    if len(sys.argv) != 4:
        raise ValueError("Usage: python scrape.py <scrape_type> <artist_name> <max_songs>")

    scrape_type, artist_name, max_songs = sys.argv[1:4]
    assert scrape_type in ("genius", "open-lyrics"), "Scrape type must be either 'genius' or 'open-lyrics'"
    assert (max_songs.isdigit()) and (int(max_songs)>0), "Maximum number of songs must be an integer"

    # do scraping & processing:
    if scrape_type == "genius":
        # scrape lyrics from Genius.com:
        Genius_key = ENV_VAR['Genius_key']   # load Genius API key from .env file
        set_loggings(level="info", func_name="Genius song Scraper")
        genius_scraper = GeniusScraper(artist_name, max_songs=int(max_songs), Genius_key=Genius_key)
        lyrics_raw = genius_scraper.raw_dict
        # process the lyrics:
        lyric_processor = LyricProcessor(lyrics_raw, scrape_type="genius", 
                                         check_invalid_title=True, check_word_cnt=True, 
                                         check_lyric_similarity=False, similarity_threshold=0.6)  # not necessary, this step is time consuming
        lyric_processor.save(genius_scraper.save_directory)   # save processed lyrics at same directory
    else:  # scrape_type == "open-lyrics":
        set_loggings(level="info", func_name="Open-Lyrics Scraper")
        # add this part after implementing open-lyrics scraper

if __name__ == '__main__':
	main()      # the func to run