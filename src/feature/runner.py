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

def main():                   # eg. python src/feature/runner.py genius
    # collect system args:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python src/feature/runner.py <'genius' or 'open-lyrics'>")

    data_type = sys.argv[1]
    assert data_type in ("genius", "open-lyrics"), "Scrape type must be either 'genius' or 'open-lyrics'"

    # do feature extraction:
    if data_type == "genius":
        genius_loader = LoadGenius(file_name = "lyrics_processed.parquet")
        df_merge = pd.DataFrame()   # buffer for final output
        for file_path in genius_loader.latest_dirs:
            emotion_classifier = EmotionClassifier(parquet_path = file_path, save=True)   # do classification
            df_merge = pd.concat([df_merge, emotion_classifier.df], ignore_index=True, axis=0)
        save_path = os.path.join("data/genius/MERGED", f"lyrics_{get_timestamp()}")
        save_path = clean_file_name(save_path, is_path=True)
        df_merge.to_parquet(save_path)
    else:  # scrape_type == "open-lyrics":
        # add this part after implementing open-lyrics scraper
        pass



if __name__ == '__main__':
	main()      # the func to run