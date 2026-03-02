# coding: utf-8
import os
import logging

def create_logger(logging_dir: str = None) -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout. Only the main process logs.

    Args:
        logging_dir (str): The directory to save the log file.

    Returns:
        logging.Logger: The logger.
    """
    additional_args = dict()
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        additional_args["handlers"] = [
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ]
    
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        **additional_args,
    )
    
    logger = logging.getLogger(__name__)
    if logging_dir is not None:
        logger.info("Experiment directory created at %s", logging_dir)
    
    return logger

