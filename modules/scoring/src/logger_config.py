import logging
import logging.config


def get_logger(log_dir, logger_name="logs"):
    logger_name = logger_name.lower()
    dict_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "myFormatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            logger_name: {
                "handlers": ["default"],
                "level": "INFO",
            }
        },
        "formatters": {
            "myFormatter": {
                "format": f"{logger_name}: %(asctime)s - %(levelname)s - %(message)s"
            }
        },
    }
    logging.config.dictConfig(dict_log_config)
    return logging.getLogger(logger_name)
