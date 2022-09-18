import uuid

import os
from bson.objectid import ObjectId

from .constants import TrainingStatus
from .config import settings

from .app import database

models_collection = database.get_collection("ml_models_v2")
training_file_path = settings.training_file_path
model_path = settings.model_path


# Retrieve all model information in the database
async def retrieve_models():
    models = []
    async for model in models_collection.find():
        models.append(model)
    return models


# Add a new model into to the database
async def add_model(name) -> dict:
    model_data = dict()
    model_data["name"] = name
    model_data["versions"] = dict()
    version_status = dict()
    version_status["version"] = uuid.uuid4().hex
    version_status["status"] = TrainingStatus.CREATED

    model_data["versions"][version_status["version"]] = version_status

    model = await models_collection.insert_one(model_data)

    new_model = await models_collection.find_one({"_id": model.inserted_id})
    new_model["training_dataset_path"] = os.path.join(training_file_path, str(model.inserted_id))
    new_model["model_path"] = os.path.join(model_path, str(model.inserted_id))

    models_collection.update_one({'_id': model.inserted_id}, {"$set": new_model}, upsert=False)
    return await models_collection.find_one({"_id": model.inserted_id})


# add new version for existing model
async def add_model_version(model) -> dict:
    version_status = dict()
    version_status["version"] = uuid.uuid4().hex
    version_status["status"] = TrainingStatus.CREATED
    model["versions"].update({version_status["version"]: version_status})
    await models_collection.update_one({'_id': ObjectId(model["_id"])}, {"$set": model}, upsert=False)
    return await models_collection.find_one({'_id': ObjectId(model["_id"])})


# update status of a given model id and version
async def update_status(id, version, status):
    model = await models_collection.find_one({"_id": ObjectId(id)})
    if model is None:
        print("model with {} not found".format(id))
    model["versions"][version]["status"] = status
    if status == TrainingStatus.TRAINED:
        model["active_version"] = version
    await models_collection.update_one({'_id': ObjectId(id)}, {"$set": model}, upsert=False)
    await models_collection.find_one({"_id": ObjectId(id)})


# retrieves model by id
async def get_model_by_id(id: ObjectId, version: str):
    model = await models_collection.find_one({"_id": id})
    if model is None:
        print("model with id {} not found".format(str(id)))
        return None

    if version not in model["versions"]:
        print("model with version {} not found".format(str(version)))
        return None
    return model


# get latest model with trained version
## todo :: find latest model
async def get_latest_model() -> dict:
    return await models_collection.find_one({"active_version": {"$exists":True}})


async def find_model_by_name(name) -> dict:
    return await models_collection.find_one({"name": name})
