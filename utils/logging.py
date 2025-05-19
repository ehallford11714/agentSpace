import structlog
import logging
from typing import Optional

def setup_logging(level: str = 'INFO') -> None:
    """
    Set up structured logging
    
    Args:
        level (str): Logging level
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[structlog.stdlib.ProcessorFormatter.wrap_for_formatter()],
    )
    
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger
    
    Args:
        name (str): Logger name
        
    Returns:
        structlog.BoundLogger: Structured logger instance
    """
    return structlog.get_logger(name)

def add_context(logger: structlog.BoundLogger, **kwargs) -> structlog.BoundLogger:
    """
    Add context to a logger
    
    Args:
        logger (structlog.BoundLogger): Logger instance
        **kwargs: Context variables
        
    Returns:
        structlog.BoundLogger: Logger with added context
    """
    return logger.bind(**kwargs)

def setup_file_logging(log_file: str, level: str = 'INFO') -> None:
    """
    Set up file logging
    
    Args:
        log_file (str): Path to log file
        level (str): Logging level
    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    ))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

def setup_console_logging(level: str = 'INFO') -> None:
    """
    Set up console logging
    
    Args:
        level (str): Logging level
    """
    handler = logging.StreamHandler()
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
    ))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
