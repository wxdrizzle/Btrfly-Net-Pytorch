import logging
import os
import sys
import colorlog

log_colors = {
    'DEBUG': 'white',
    'INFO': '',
    'WARNING': 'cyan',
    'ERROR': 'red',
    'CRITICAL': 'purple',
}


def setup_colorful_logger(name, save_dir=None, format="only_message"):
    """
    set up colorful logger (colors are defined in log_colors[type: dict])
    :param name: logger's name
    :param save_dir: save log info to a txt file
    :param format: only message will be printed if it is a str named "only_message"
    :return: the logger
    """
    logger = colorlog.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = colorlog.StreamHandler(stream=sys.stdout)

    if format != "only_message":
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s[%(asctime)s] %(name)s: %(message)s",
            datefmt='%m-%d %H:%M:%S',
            log_colors=log_colors,
        )
    else:
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(message)s",
            datefmt='%m-%d %H:%M:%S',
            log_colors=log_colors,
        )

    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    if save_dir:
        fh = logging.FileHandler(save_dir)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# draw a separator
def separator(logger):
    logger.info(
        "———————————————————————————————————————————————————————————————————————————————————————————————————————")
