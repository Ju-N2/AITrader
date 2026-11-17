import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aitrader.config.settings import get_settings


def setup_logger(name: str = "aitrader") -> logging.Logger:
    settings = get_settings()

    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level.upper())

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_dir / "aitrader.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
