'''
functions to setup logging for the application.

setup_logging(
    level=logging.DEBUG,            # Control Level: DEBUG, INFO, WARNING, ERROR
    log_to_console=True,            # Toggle Console output
    log_file_path="logs/app.log"    # Set file path to save logs (or None to disable)
)

'''
import logging
import sys
from pathlib import Path

def setup_logging(
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_file_path: str | Path | None = None
) -> None:
    """
    Configures the root logger. All other files using logging.getLogger(__name__)
    will inherit these settings automatically.
    
    :param level: The logging level (e.g., logging.DEBUG, logging.INFO, etc.)
    :param log_to_console: If True, outputs logs to the terminal/console.
    :param log_file_path: If a path is provided, writes logs to this file.
    """
    # 1. Clear any existing handlers on the root logger to prevent duplicate logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(level)

    # 2. Define a clean, professional formatting style
    # [Timestamp] [Level] [Source File:Line] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. Setup Console/Terminal Output
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 4. Setup File Output
    if log_file_path:
        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Create logs folder if missing
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)