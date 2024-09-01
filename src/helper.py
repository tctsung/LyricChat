import logging
from datetime import datetime
import os
import re

def set_loggings(level=logging.INFO, func_name=''):
	"""
	TODO: set logging levels
	"""
	if isinstance(level, str):
		level = level.upper()
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
    # param path (str): path to one of the dir in data/genius/
    directory = os.path.dirname(path)
    with open(os.path.join(directory, 'LyricScraper.meta'), 'r') as fp:
        for line in fp:
            if line.startswith('Artist name: '):
                return line.split(':')[1].strip()
def clean_file_name(file_name):
	# TODO: standardize file names
	name_split = file_name.rsplit('.', 1)    # right split & max split is one
	file_name = name_split[0]                # get file name without extension
	# do processing:
	cleaned_name = re.sub(r'\s+', ' ', file_name.upper())
	cleaned_name = re.sub(r'[\s:-]', '_', cleaned_name)
	# add file extension back
	if len(name_split) > 1:
		cleaned_name += '.' + name_split[1]   
	return cleaned_name
def find_files(directory_path = ".", file_extension=".json"):
	# TODO: find all files in a directory with specific file extension
    file_paths = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths
def format_timedelta(td):
    # Ensure we're working with a positive timedelta
    td = abs(td)
    # Extract hours, minutes, and seconds
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d} hr {minutes:02d} min {seconds:02d} sec"
## Suppress stdout:
# import os
# import contextlib
# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#     "Your code..."