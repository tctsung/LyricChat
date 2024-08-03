import json
import re
import logging
import helper
import os
def main():
    # set parameters:
    logging_level = 'INFO'
    irrelevant_words = [r"See Imagine Dragons LiveGet tickets as low as \$\d+"]

    # get an iterator of all json files at directory:
    directory_path = "data\lyrics"
    helper.find_files(directory_path, file_extension=".json")
    # do preprocessing:
    helper.set_loggings(level=logging_level, func_name='LyricPreprocessor')
    for file_path in helper.find_files(directory_path, file_extension=".json"):
        if not os.path.basename(file_path).endswith("_preprocess.json"):   # exclude preprocessed files
            logging.info("Start preprocessing file `%s` at directory `%s`", 
                         os.path.basename(file_path), 
                         os.path.dirname(file_path)
                         )
            LyricPreprocessor(file_path, irrelevant_words, save=True)

class LyricPreprocessor:
    """
    TODO: preprocess the scraped lyrics to reduce noise and remove irrelevant words
    """
	# commonly seen irrelevant words in Genius lyrics:
    default_irrelevant_words = [
        r"You might also like",
        r"\d*Embed$"
    ]
    def __init__(self, file_path: str, irrelevant_words: list, save = False):
        """
        param file_path (str): the path of the scraped json file
        param irrelevant_words (list): a list of irrelevant words to remove from the lyrics
        param save (bool): whether to save the preprocessed lyrics to the same directory
        """
        # check file_path is json file:
        assert file_path.endswith(".json"), "file_path must be json file!"

        # set parameters:
        self.file_path = file_path
        self.irrelevant_words = LyricPreprocessor.default_irrelevant_words
        if len(irrelevant_words) > 0:
            self.irrelevant_words.extend(irrelevant_words)
        
        # run preprocessing:
        self.load_lyrics()    # load lyrics
        self.preprocess()     # preprocessing

        # save file:
        if save:
            self.save()
    def load_lyrics(self):
        # TODO: load the scraped lyrics from src\lyric_scrapping\scrape.py
        with open(self.file_path) as f:
            self.lyrics_raw = json.load(f)
    def preprocess(self, min_word_cnt=30):
        # TODO: preprocess the lyrics
        removed_songs = []
        lyrics_clean = {}
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

            # turn multiple new lines into single new line to split chunks:
            lyric_process = re.sub(r'\n\n\n+', '\n\n', lyric_process)

            # remove leading and trailing spaces:
            lyric_process = lyric_process.strip()

            # keep only lyrics that have len > min_len
            if len(lyric_process.split()) < min_word_cnt:
                removed_songs.append(title)
            else:
                lyrics_clean[title] = lyric_process
        if len(removed_songs) > 0:
            logging.warning("%d songs were removed due to inadquate word count", len(removed_songs))
        self.lyrics_clean = lyrics_clean
        self.removed_songs = removed_songs
    def save(self):
        # TODO: save the preprocessed lyrics to the same directory
        directory_name = os.path.dirname(self.file_path)
        file_name = os.path.basename(self.file_path)
        self.save_file = file_name[:-5] + "_preprocess" + ".json"
        self.save_path = os.path.join(directory_name, self.save_file)
        with open(self.save_path, 'w') as f:
            json.dump(self.lyrics_clean, f, indent=4)
        logging.info("File `%s` saved at directory `%s`", self.save_file, directory_name)
         
                
             
if __name__ == '__main__':
	main()      # the func to run