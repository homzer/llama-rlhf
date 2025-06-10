import inspect
import os.path
import sys
from datetime import datetime


class Logger:
    def __init__(self, log_dir: str = "."):
        os.makedirs(log_dir, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_dir, "output.log"), "w", encoding="utf-8")

    def write(self, message: str):
        if message.strip():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            frame = inspect.currentframe().f_back
            file_name = os.path.basename(frame.f_code.co_filename)
            line_number = frame.f_lineno
            formatted_message = f"[{current_time} | {file_name}:{line_number}] {message}"
            self.terminal.write(formatted_message)
            self.log.write(formatted_message)
        else:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.flush()


def init_logger(log_dir: str = "."):
    sys.stdout = Logger(log_dir=log_dir)
