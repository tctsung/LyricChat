import logging
from datetime import datetime
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
def get_artist_name(path):
    # TODO: get artist name from LyricScraper.meta
    # param path (str): path to one of the dir in data/genius/
    directory = os.path.dirname(path)
    with open(os.path.join(directory, 'LyricScraper.meta'), 'r') as fp:
        for line in fp:
            if line.startswith('Artist name: '):
                return line.split(':')[1].strip()