import logging
from datetime import datetime
import re
import os
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

def clean_file_name(file_name):
	cleaned_name = re.sub(r'\s+', ' ', file_name.upper())
	cleaned_name = re.sub(r'[\s:-]', '_', cleaned_name)
	return cleaned_name

def find_files(directory_path = ".", file_extension=".json"):
	# TODO: find all files in a directory with specific file extension
    file_paths = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    
    return file_paths
## Suppress stdout:
# import os
# import contextlib
# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#     "Your code..."