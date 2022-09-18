import motor.motor_asyncio
import time
import os
from bson.objectid import ObjectId

from server.constants import TrainingStatus

MONGO_DETAILS = "mongodb://localhost:27017"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)

database = client.ml_serving

models_collection = database.get_collection("ml_models")
history_collection = database.get_collection("history")
destination_file_path = "C:\\Users\kansh\\train_data\\"
evaluate_path = "C:\\Users\\kansh\\evaluate_models"


def models_helper(model) -> dict:
    return {
        "id": str(model["_id"]),
        "name": model["name"],
        "version": model["version"],
        "status": str(TrainingStatus(model["status"])),
        "training_dataset_path": model["training_dataset_path"],
        "model_path": model["model_path"]
    }

def evaluation_helper(data) -> dict:
    return {
        "id": str(data["_id"]),
        "model_id": str(data["model_id"]),
        "evaluation_path": data["evaluation_path"],
        "truth": data["truth"],
        "result": (data["result"])
    }


# Retrieve all model information in the database
async def retrieve_models():
    models = []
    async for model in models_collection.find():
        models.append(models_helper(model))
    return models


# Add a new model into to the database
async def add_model(model_data) -> dict:
    model_data["status"] = TrainingStatus.CREATED
    model_data["created_on"] = int(time.time())

    model = await models_collection.insert_one(model_data)
    new_model = await models_collection.find_one({"_id": model.inserted_id})
    print(new_model)
    new_model["training_dataset_path"] = destination_file_path + str(model.inserted_id)
    os.makedirs(new_model["training_dataset_path"])

    model_dir = "C:\\Users\\kansh\\train_models\\" + str(model.inserted_id)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    new_model["model_path"] = model_dir

    models_collection.update_one({'_id': model.inserted_id}, {"$set": new_model}, upsert=False)
    new_model = await models_collection.find_one({"_id": model.inserted_id})
    return models_helper(new_model)


async def update_status(id, status):
    model = await models_collection.find_one({"_id": ObjectId(id)})
    if model is None:
        print("model with {} not found".format(id))
    model["status"] = status
    await models_collection.update_one({'_id': ObjectId(id)}, {"$set": model}, upsert=False)
    new_model = await models_collection.find_one({"_id": ObjectId(id)})
    print(new_model)


async def get_model_by_id(id) -> dict:
    model = await models_collection.find_one({"_id": ObjectId(id)})
    if model is None:
        return None
    return models_helper(model)

async def get_latest_model() -> dict:
    model = await models_collection.find_one({"status": int(TrainingStatus.TRAINED)})
    return models_helper(model)


async def add_evaluation(evaluation_data) -> dict:
    evaluation_data["created_on"] = int(time.time())
    evaluation_data["result"] = dict()

    data = await history_collection.insert_one(evaluation_data)
    new_data = await history_collection.find_one({"_id": data.inserted_id})
    new_data["evaluation_path"] = os.path.join(evaluate_path, str(data.inserted_id))
    os.makedirs(new_data["evaluation_path"])
    os.makedirs(os.path.join(new_data["evaluation_path"], "data"))

    await history_collection.update_one({'_id': data.inserted_id}, {"$set": new_data}, upsert=False)
    new_model = await history_collection.find_one({"_id": data.inserted_id})
    return evaluation_helper(new_model)

async def update_evaluation(evaluation_data) -> dict:
    print(evaluation_data)
    data = await history_collection.update_one({"_id": ObjectId(evaluation_data["id"])}, {"$set": evaluation_data})
    new_data = await history_collection.find_one({"_id": ObjectId(evaluation_data["id"])})
    return evaluation_helper(new_data)

async def fetch_all_history() -> list:
    data = history_collection.find({})
    result = list()
    async for d in data:
        result.append(evaluation_helper(d))
    return result

async def fetch_history_for_model(model_id) -> list:
    data = history_collection.find({"model_id": ObjectId(model_id)})
    result = list()
    async for d in data:
        result.append(evaluation_helper(d))
    return result