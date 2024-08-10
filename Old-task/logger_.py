import logging
from pathlib import Path
import sys
import traceback


__all__ = [
    'create_logger',
    'log_exception',
]


# Configure logging for file logging
logging.basicConfig(
    filename=str(Path(__file__).parent.parent.parent / 'public' / 'logs'),
    filemode='a',
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    )


# Create logger
def create_logger(tag=None):
    if tag is None:
        return logging.getLogger('ai.api')
    return logging.getLogger(f"ai.api.{tag}")


# Log exception
def log_exception(logger, e):
    logger.error(f"EXCEPTION: {repr(e)}")

    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)

    logger.error(f"- Exception type: {ex_type.__name__}")
    logger.error(f"- Exception message: {ex_value}")
    logger.error(f"- Stack trace:")
    for i, trace in enumerate(trace_back):
        logger.error(f"{'-' * (i + 2)} In {trace[0]}, line {trace[1]}, in function {trace[2]}: '{trace[3]}'")


# Add stream handler to logger for console logging
stream_handler = logging.StreamHandler(sys.stdout)
create_logger().addHandler(stream_handler)
