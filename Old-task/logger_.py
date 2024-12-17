import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import traceback

class Logger:
    def __init__(self, log_path=None):
        self.log_path = log_path or str(Path(__file__).parent.parent.parent / 'public' / 'logs' / 'app.log')
        self._setup_logging()
    
    def _setup_logging(self):
        # Base configuration
        logging.basicConfig(
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            level=logging.INFO
        )
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            self.log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        
        # Root logger configuration
        root_logger = logging.getLogger('ai.api')
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    @staticmethod
    def get_logger(tag=None):
        if tag is None:
            return logging.getLogger('ai.api')
        return logging.getLogger(f"ai.api.{tag}")
    
    @staticmethod
    def log_exception(logger, exception):
        logger.error(f"EXCEPTION: {repr(exception)}")
        
        ex_type, ex_value, ex_traceback = sys.exc_info()
        trace_back = traceback.extract_tb(ex_traceback)
        
        stack_trace = [
            f"Exception type: {ex_type.__name__}",
            f"Exception message: {ex_value}",
            "Stack trace:"
        ]
        
        for i, trace in enumerate(trace_back):
            stack_trace.append(
                f"{'-' * (i + 2)} In {trace[0]}, line {trace[1]}, "
                f"in function {trace[2]}: '{trace[3]}'"
            )
        
        logger.error('\n'.join(stack_trace))

# Initialize logger
logger = Logger()
create_logger = logger.get_logger
log_exception = logger.log_exception

__all__ = ['create_logger', 'log_exception']