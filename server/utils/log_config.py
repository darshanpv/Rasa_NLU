import time
from pathlib import Path

LOG_LEVEL: str = "INFO"
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": '%(asctime)s %(filename)s %(levelname)s:%(message)s',
            "use_colors": True,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "filelog_default": {
            "formatter": "default",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": Path(f'logs/{time.strftime("%Y-%m-%d", time.localtime())}.log'),
            "encoding": "utf8",
            "backupCount": 5,
            "when": "midnight",
        },
        "filelog_access": {
            "formatter": "access",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": Path(f'logs/{time.strftime("%Y-%m-%d", time.localtime())}.log'),
            "encoding": "utf8",
            "backupCount": 7,
            "when": "midnight",
        },
    },
    "loggers": {
        "": {"handlers": ["default", "filelog_default"], "level": LOG_LEVEL},
        "uvicorn.error": {"level": LOG_LEVEL},
        "uvicorn.access": {"handlers": ["access", "filelog_access"], "level": LOG_LEVEL, "propagate": False},
    },
}