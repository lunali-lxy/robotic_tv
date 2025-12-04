# logger_utils.py
import logging
import os
from datetime import datetime

def setup_logger(log_name='TVTracker', log_file='app.log', level=logging.INFO):
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warnning there is no file handler: {e}")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    logger.info(f"Logger '{log_name}' has been configured, outputting to {log_file}.")
    return logger
log_dir = 'logs'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_LOG_FILE = os.path.join(log_dir, f'run_{timestamp}.log')