import logging
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_FILE, MAX_FILE_SIZE_MB


def setup_logger():
    logger = logging.getLogger('dog_detector')
    logger.setLevel(LOG_LEVEL)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=max_bytes,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()
