from datetime import datetime

from pydantic import BaseModel, Field
from server.constants import TrainingStatus

class MlModelSchema(BaseModel):
    name: str = Field(...)
    version: str = Field(...)
    created_on: datetime = Field(...)
    model_path: str = Field(...)
    status: TrainingStatus = Field(...)
    training_dataset_path: str = Field(...)

class MlModelCreate(BaseModel):
    name: str = Field(...)
    version: str = Field(...)

def ResponseModel(data, message, code = 200):
    return {
        "data": [data],
        "code": code,
        "message": message,
    }


def ErrorResponseModel(error, code, message):
    return {"error": error, "code": code, "message": message}

class Base(BaseModel):
    name: str
    truth: str