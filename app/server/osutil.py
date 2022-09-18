import os
from pathlib import Path

class OsUtil:
    def __init__(self):
        self.training_data_dir_path = os.environ['HOME'] + "training_data"
        self.training_data_dir = Path("training_data_dir_path").mkdir(parents=True, exist_ok=True)
