from aitrader.core.logger import setup_logger


def test_logger_creation():
    logger = setup_logger("test_logger")

    assert logger.name == "test_logger"
    assert logger.level > 0

    logger.info("AITrader logger test")
