from fastapi import APIRouter
from fastapi import UploadFile, File, Depends, Form
import aiofiles
from fastapi.encoders import jsonable_encoder
import os
from fastapi.concurrency import run_in_threadpool
from bson.objectid import ObjectId

from server.database import (
    retrieve_models,
    add_model,
    get_model_by_id,
    update_status,
    add_evaluation,
    update_evaluation,
    get_latest_model,
    fetch_history_for_model,
    fetch_all_history
)
from server.models.ml_model import (
    ResponseModel,
    MlModelCreate,
    Base
)
from server.constants import TrainingStatus
from ml_models.mobileNetModel import MobileNetBasedModel
from model_store.ModelStore import model_store

from fastapi import BackgroundTasks

router = APIRouter()

@router.get("/metadata", response_description="Models data from the database")
async def retrieve_all_models_data():
    models = await retrieve_models()
    return ResponseModel(models, "All Models from DB")


@router.post("/train", response_description="create a model metadata for training")
async def train_model(req: MlModelCreate):
    req_json = jsonable_encoder(req)
    model = await add_model(req_json)
    return ResponseModel(model, "created a model for training, start training now")

@router.put("/train/{model_id}/{label}", response_description="add images to a model for training with label")
async def train_model(model_id,
                      label,
                      files: list[UploadFile] = File(default=..., description="multiple file with name as type")):
    model = await get_model_by_id(model_id)
    if model is None:
        print("model not found")
        return ResponseModel(None, "model_id does not exist", 400)

    destination_file_dir = model["training_dataset_path"] + "\\" + label + "\\"
    if not os.path.isdir(destination_file_dir):
        os.makedirs(destination_file_dir)

    for file in files:
        destination_file_path = model["training_dataset_path"] + "\\" + label + "\\" + file.filename
        async with aiofiles.open(destination_file_path, 'wb') as out_file:
            while content := await file.read(1024):  # async read file chunk
                await out_file.write(content)  # async write file chunk

    return ResponseModel({"Result": "OK", "filenames": [file.filename for file in files]}, "Files saved to db")

@router.post("/train/submit/{model_id}")
async def train_model(model_id, background_tasks: BackgroundTasks):
    model = await get_model_by_id(model_id)
    if model is None:
        print("model not found")
        return ResponseModel(None, "model_id does not exist", 400)

    if model["status"] == str(TrainingStatus.TRAINED):
        return ResponseModel({"Result": "OK"}, "Model already trained")

    if model["status"] == str(TrainingStatus.TRAINING_IN_PROGRESS):
        return ResponseModel({"Result": "OK"}, "Training already in progress", 208)

    await update_status(model["id"], TrainingStatus.TRAINING_IN_PROGRESS)

    background_tasks.add_task(train_model, model)

    return ResponseModel({"Result": "OK"}, "Model training started", 202)

async def train_model(model):
    ml_model = MobileNetBasedModel(model["training_dataset_path"], model["model_path"])
    try:
        await run_in_threadpool(lambda: ml_model.train())
        await update_status(model["id"], TrainingStatus.TRAINED)
        await model_store.load(model)
    except Exception as e:
        print(e)
        await update_status(model["id"], TrainingStatus.FAILED)


evaluate_path = "C:\\Users\\kansh\\evaluate_models"

@router.post("/evaluate/{model_id}")
async def evaluate_model(model_id,
                         files: list[UploadFile] = File(default=..., description="multiple file with name"),
                         filenames: str = Form(...),
                         truths: str = Form(...)
                         ):
    filenames = filenames.split(",")
    truths = truths.split(",")
    model = await get_model_by_id(model_id)
    if model is None:
        print("model not found")
        return ResponseModel(None, "model_id does not exist", 400)

    evaluation_data = dict()
    evaluation_data["model_id"] = ObjectId(model["id"])
    truth_data = dict()
    for file, truth in zip(filenames, truths):
        truth_data[file] = truth
    evaluation_data["truth"] = truth_data

    evaluation_data = await add_evaluation(evaluation_data)
    for file in files:
        destination_file_path = os.path.join(evaluation_data["evaluation_path"], "data", file.filename)
        async with aiofiles.open(destination_file_path, 'wb') as out_file:
            while content := await file.read(1024):  # async read file chunk
                await out_file.write(content)  # async write file chunk

    ml_model = model_store.get_running_model(model)
    if ml_model is None:
        return ResponseModel(None, "Model Not trained yet", 500)

    preds = ml_model.predict(evaluation_data["evaluation_path"])

    result = dict()
    for file, pred in zip(filenames, preds):
        result[file] = float(pred)
    evaluation_data["result"] = result
    evaluation_data = await update_evaluation(evaluation_data)
    print(evaluation_data)
    return ResponseModel(evaluation_data, "Evaluation Done", 202)


@router.post("/predict")
async def evaluate_model(file: UploadFile = File(default=..., description="file with name")):
    model = await get_latest_model()

    evaluation_data = dict()
    evaluation_data["model_id"] = ObjectId(model["id"])
    evaluation_data["truth"] = dict()

    evaluation_data = await add_evaluation(evaluation_data)
    destination_file_path = os.path.join(evaluation_data["evaluation_path"], "data", file.filename)
    async with aiofiles.open(destination_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read file chunk
            await out_file.write(content)  # async write file chunk

    ml_model = model_store.get_running_model(model)
    if ml_model is None:
        return ResponseModel(None, "Model Not trained yet", 500)

    pred = ml_model.predict(evaluation_data["evaluation_path"])

    result = dict()
    result[file.filename] = float(pred)
    evaluation_data["result"] = result
    evaluation_data = await update_evaluation(evaluation_data)
    print(evaluation_data)
    return ResponseModel(evaluation_data, "prediction Done", 202)


@router.get("/history")
async def evaluate_model(model_id: str = None):
    if model_id is None:
        evaluation_data_list = await fetch_all_history()
    else:
        evaluation_data_list = await fetch_history_for_model(model_id)

    return ResponseModel(evaluation_data_list, "prediction Done", 202)