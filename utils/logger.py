import logging
import os
import sys
from datetime import datetime


class Logger:
    def __init__(self, name, log_dir="logs", is_main_process=True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.log_dir = log_dir
        self.is_main_process = is_main_process

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)

    def info(self, msg):
        if self.is_main_process:
            self.logger.info(msg)

    def warning(self, msg):
        if self.is_main_process:
            self.logger.warning(msg)

    def error(self, msg):
        # Error messages should always be logged, even from non-main processes
        self.logger.error(msg)

    def log_metrics(self, metrics):
        """
        Log a dictionary of metrics.
        """
        if self.is_main_process:
            msg = " | ".join([f"{k}: {v}" for k, v in metrics.items()])
            self.logger.info(msg)
