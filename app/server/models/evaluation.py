from datetime import datetime

from pydantic import BaseModel, Field
from bson.objectid import ObjectId


class Evaluation(BaseModel):
    model_id: ObjectId = Field(...)
    created_on: datetime = Field(...)
    evaluation_path: str = Field(...)
    truth: dict = Field(...)
    result: dict = Field(...)