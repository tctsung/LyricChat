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
def clean_file_name(file_name, is_path):
	# TODO: standardize file names
	# param file_name (str): name of the file
	# param is_path (bool): if the input is a path (includes \ and /, which represents a path or not)
	name_split = file_name.rsplit('.', 1)    # right split & max split is one
	file_name = name_split[0]                # get file name without extension
	# do processing:
	if is_path:      # ignore / and \ if input is a path
		cleaned_name = re.sub(r'[\s:-]', ' ', file_name)   # .upper()
		cleaned_name = re.sub(r'\s+', '_', cleaned_name.strip())
	else:
		cleaned_name = re.sub(r'[\s:/\\-]', ' ', file_name)   # .upper()
		cleaned_name = re.sub(r'\s+', '_', cleaned_name.strip())
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
def latest_file(directory):
    # TODO: get the latest file from the provided directory
    # the latest file must contain get_timestamp() output in the end of file name
	files = os.listdir(directory)   # get all files
	latest_timestamp, latest_file = None, None
	for fl in files:
		fl_no_ext = fl.rsplit('.', 1)[0]        # get file name without extension
		parts = fl_no_ext.split('_')
		try:
			timestamp = datetime.strptime('_'.join(parts[-6:]), '%Y_%m_%d_%H_%M_%S')
			if latest_timestamp is None or timestamp > latest_timestamp:
				latest_timestamp = timestamp
				latest_file = fl
		except:
			logging.warning("Cannot parse timestamp in file: %s", fl)
	return os.path.join(directory, latest_file)

## Suppress stdout:
# import os
# import contextlib
# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#     "Your code..."