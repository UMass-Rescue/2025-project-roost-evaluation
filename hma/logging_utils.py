import logging
import os
from datetime import datetime
from pathlib import Path

# Setup logging
logger = None
log_file = None

def setup_logging(test_run_type="test"):
    """Setup file logging and minimal terminal output"""
    global logger, log_file

    # Create logs directory
    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    logs_dir = output_root / "test_run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename: TestRunType_Date_Timestarted
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{test_run_type}_{timestamp}.log"
    log_file = logs_dir / log_filename

    # Setup logger
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear any existing handlers

    # File handler - all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler - only WARNING and above, minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Direct logger access since we just initialized it
    logger.info(f"Logging to: {log_file}")
    # Flush to ensure the log file is created and accessible
    file_handler.flush()
    return logger

def get_log_file():
    """Path of the current log file, or None if setup_logging hasn't been called."""
    return log_file

# Fallback logger if setup_logging hasn't been called
def get_logger():
    """Get logger - uses parent's logger if available, otherwise NullHandler"""
    global logger
    if logger is None:
        logger = logging.getLogger("evaluation")
        logger.setLevel(logging.DEBUG)
        # Check if logger already has handlers (from setup_logging in parent)
        if not logger.handlers:
            # No parent logger - use NullHandler (tests run directly, not as subprocess)
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
    return logger

# Use get_logger() wrapper to ensure logger is always available
def _log_debug(msg):
    get_logger().debug(msg)
def _log_info(msg):
    get_logger().info(msg)
def _log_warning(msg):
    get_logger().warning(msg)
def _log_error(msg):
    get_logger().error(msg)
