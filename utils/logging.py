import logging
import os


def init_logging(save_path, save_file='log.txt'):
    logfile = os.path.join(save_path, save_file)

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

def log_and_print(line):
    print(line)
    logging.info(line)
