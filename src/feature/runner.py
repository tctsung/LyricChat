import os
import sys

# set working directory to LyricChat repo root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.chdir('../..')   

# load self-written functions:
sys.path.append("src/")     
from helper import *
sys.path.append("src/feature/")
from extract_hf import *  

# get secret tokens:
from dotenv import dotenv_values
ENV_VAR = dotenv_values(".env")

def main():                      # eg. python src/feature/runner.py genius
    # collect system args:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python src/feature/runner.py <'genius' or 'open-lyrics'>")

    scrape_type = sys.argv[1]
    assert scrape_type in ("genius", "open-lyrics"), "Scrape type must be either 'genius' or 'open-lyrics'"

    # do feature extraction:
    if scrape_type == "genius":    # genius data
        set_loggings(level="info", func_name="Emotion Classifier for Genius.com")
        genius_loader = LoadGenius(file_name = "lyrics_processed.parquet")
        logging.info("Identified %d artists", len(genius_loader.latest_dirs))
        df_merge = pd.DataFrame()   # buffer for final output
        for artist, file_path in genius_loader.latest_dirs.items():
            logging.info("Processing %s", artist)
            emotion_classifier = EmotionClassifier(parquet_path = file_path, save=True)   # do classification
            df_merge = pd.concat([df_merge, emotion_classifier.df.copy()], ignore_index=True, axis=0)
            del emotion_classifier   # save memory
        save_path = os.path.join("data/genius/MERGED", f"lyrics_{get_timestamp()}")
        save_path = clean_file_name(save_path, is_path=True)
        df_merge.to_parquet(save_path + ".parquet")
        df_merge.to_csv(save_path + ".csv",encoding='utf-8-sig' , index=False)
        logging.info("Files saved to %s in parquet and csv format", save_path)
    else:                        # scrape_type == "open-lyrics":
        # add this part after implementing open-lyrics scraper
        pass



if __name__ == '__main__':
	main()      # the func to run