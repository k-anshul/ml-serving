from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    home = os.path.expanduser('~')
    print("home {}".format(home))
    app_name: str = "Ml Serving API"
    mongo_details: str = "mongodb://localhost:27017"
    training_file_path: str = os.path.join(home, "train_data")
    evaluate_file_path: str = os.path.join(home, "evaluate_data")
    model_path: str = os.path.join(home, "models")

settings = Settings()