import logging
import os 
from datetime import datetime

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): The name of the logger (usually the module or file name).
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)
    return logger

if __name__ == '__main__':
    logger = get_logger('talha')
    logger.info('Log file created')