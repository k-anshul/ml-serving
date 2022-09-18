from fastapi import FastAPI

from .routes.ml_routes import router as ModelRouter
from .config import settings
from.osutil import create_dir_if_not_exist
app = FastAPI()

app.include_router(ModelRouter, tags=["Models"], prefix="/app")

@app.on_event("startup")
async def startup_event():
    create_dir_if_not_exist(settings.training_file_path)
    create_dir_if_not_exist(settings.evaluate_file_path)
    create_dir_if_not_exist(settings.model_path)
