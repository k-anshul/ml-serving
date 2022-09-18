import os
from .app import database
from .config import settings
from bson import ObjectId

history_collection = database.get_collection("history_v2")
evaluate_path = settings.evaluate_file_path


async def add_evaluation(evaluation_data) -> dict:
    evaluation_data["result"] = dict()
    data = await history_collection.insert_one(evaluation_data)
    new_data = await history_collection.find_one({"_id": data.inserted_id})
    new_data["evaluation_path"] = os.path.join(evaluate_path, str(data.inserted_id))

    await history_collection.update_one({'_id': data.inserted_id}, {"$set": new_data}, upsert=False)
    return await history_collection.find_one({"_id": data.inserted_id})


async def update_evaluation(evaluation_data) -> dict:
    await history_collection.update_one({"_id": evaluation_data["_id"]}, {"$set": evaluation_data})
    return await history_collection.find_one({"_id": evaluation_data["_id"]})


async def fetch_all_history() -> list:
    data = history_collection.find({})
    result = list()
    async for d in data:
        result.append(d)
    return result


async def fetch_history_for_model(model_id) -> list:
    data = history_collection.find({"model_id": ObjectId(model_id)})
    result = list()
    async for d in data:
        result.append(d)
    return result
