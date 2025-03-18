import logging
import logging.config


def get_logger(name: str = None):
    """
    Configure default logging system

    Parameters
    ----------
    name : str
        Logger name. If unspecified, attempts to check the call stack
        and name after calling module (file). else, called `data-science`

    Returns
    -------
    logger : Logger
        logger instance

    """

    conf = logging_configuration()

    logging.config.dictConfig(conf)
    logger = logging.getLogger(name)

    return logger


def logging_configuration():
    # Default console handler (no colors)
    simple_formatter = {
        "format": "%(asctime)s|%(levelname)7s|%(filename)25s:%(lineno)3s %(funcName)30s()| %(message)s",
        "datefmt": "%H:%M:%S",
    }

    console_handler = {
        "class": "logging.StreamHandler",
        "level": "DEBUG",
        "formatter": "simple",
        "stream": "ext://sys.stdout",
    }

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": simple_formatter},
        "handlers": {"console": console_handler},
        "root": {"level": "INFO", "handlers": ["console"]},
    }

    return config_dict
