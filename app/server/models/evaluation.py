from pydantic import BaseModel, Field
from bson import ObjectId
from .PyObjectId import PyObjectId


class Evaluation(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    model_id: PyObjectId = Field(default_factory=PyObjectId, alias="model_id")
    evaluation_path: str = Field(...)
    truth: dict = Field(...)
    model_version: str = Field(...)
    result: dict = Field(...)
    is_prediction: bool = Field(False)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
