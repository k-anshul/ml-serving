from datetime import datetime

from pydantic import BaseModel, Field
from ..constants import TrainingStatus
from bson import ObjectId
from .PyObjectId import PyObjectId
import uuid

class VersionStatus(BaseModel):
    version: uuid.UUID = Field(...)
    status: TrainingStatus = Field(...)
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {uuid.UUID: str}

class MlModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    versions: dict = Field(...)
    active_version: str = Field(default=None)
    model_path: str = Field(...)
    training_dataset_path: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class MlModelCreate(BaseModel):
    name: str = Field(...)
