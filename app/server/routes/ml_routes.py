from fastapi import APIRouter
from fastapi import UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
from fastapi.concurrency import run_in_threadpool
from bson.objectid import ObjectId
from typing import List
from ..osutil import create_dir_if_not_exist, write_to_file

from ..database import (
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
from ..models.ml_model import (
    MlModelCreate,
    MlModel
)
from ..models.evaluation import Evaluation

from ..constants import TrainingStatus
from ml_models.mobileNetModel import MobileNetBasedModel
from model_store.ModelStore import model_store

from fastapi import BackgroundTasks

router = APIRouter()


@router.get("/metadata", response_description="Models data from the database", response_model=List[MlModel])
async def retrieve_all_models_data():
    models = await retrieve_models()
    return models


@router.post("/train", response_description="create a model metadata for training", response_model=MlModel)
async def train_model(req: MlModelCreate):
    req_json = jsonable_encoder(req)
    model = await add_model(req_json)
    create_dir_if_not_exist(model["training_dataset_path"])
    create_dir_if_not_exist(model["model_path"])
    return model


async def get_model(model_id):
    try:
        object_id = ObjectId(model_id)
    except Exception as e:
        return HTTPException(status_code=400, detail=e.args)

    model = await get_model_by_id(object_id)
    if model is None:
        print("model not found")
        return HTTPException(status_code=404, detail=f"Model {id} not found")
    return model


@router.put("/train/{model_id}/{label}", response_description="add images to a model for training with label")
async def train_model(model_id,
                      label,
                      files: list[UploadFile] = File(default=..., description="multiple file with name as type")):
    model = get_model(model_id)
    destination_file_dir = os.path.join(model["training_dataset_path"], label)
    create_dir_if_not_exist(destination_file_dir)

    for file in files:
        destination_file_path = os.path.join(destination_file_dir, file.filename)
        await write_to_file(file, destination_file_path)

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=[file.filename for file in files])


@router.post("/train/submit/{model_id}", response_description="submit model for training")
async def train_model(model_id, background_tasks: BackgroundTasks):
    model = await get_model(model_id)

    if model["status"] == int(TrainingStatus.TRAINED):
        return JSONResponse(status_code=status.HTTP_208_ALREADY_REPORTED, content="already trained model")

    if model["status"] == int(TrainingStatus.TRAINING_IN_PROGRESS):
        return JSONResponse(status_code=status.HTTP_208_ALREADY_REPORTED, content="training already in progress")

    await update_status(model["_id"], TrainingStatus.TRAINING_IN_PROGRESS)

    background_tasks.add_task(train_model, model)

    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content="training started")


async def train_model(model):
    ml_model = MobileNetBasedModel(model["training_dataset_path"], model["model_path"])
    try:
        await run_in_threadpool(lambda: ml_model.train())
        await update_status(model["_id"], TrainingStatus.TRAINED)
    except Exception as e:
        print(e)
        await update_status(model["_id"], TrainingStatus.FAILED)
    model_store.load_model(model)

evaluate_path = "C:\\Users\\kansh\\evaluate_models"


@router.post("/evaluate/{model_id}", response_description="evaluate a model for given data", response_model=Evaluation)
async def evaluate_model(model_id,
                         files: list[UploadFile] = File(default=..., description="multiple file with name"),
                         filenames: str = Form(...),
                         truths: str = Form(...)
                         ):
    model = await get_model(model_id)
    if model["status"] != TrainingStatus.TRAINED:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"msg": "model not trained yet"})

    ml_model = model_store.get_running_model(model)
    if ml_model is None:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"msg": "model not trained yet"})

    filenames = filenames.split(",")
    truths = truths.split(",")

    evaluation_data = dict()
    evaluation_data["model_id"] = ObjectId(model["_id"])
    evaluation_data["is_prediction"] = False
    truth_data = dict()
    for file, truth in zip(filenames, truths):
        truth_data[file] = truth
    evaluation_data["truth"] = truth_data

    evaluation = await add_evaluation(evaluation_data)
    create_dir_if_not_exist(evaluation["evaluation_path"])
    create_dir_if_not_exist(os.path.join(evaluation["evaluation_path"], "data"))
    for file in files:
        destination_file_path = os.path.join(evaluation["evaluation_path"], "data", file.filename)
        await write_to_file(file, destination_file_path)

    preds = ml_model.predict(evaluation["evaluation_path"])

    result = dict()
    for file, pred in zip(filenames, preds):
        result[file] = float(pred)
    evaluation["result"] = result
    return await update_evaluation(evaluation)

@router.post("/predict", response_description="predict for given data", response_model=Evaluation)
async def predict(file: UploadFile = File(default=..., description="file with name")):
    model = await get_latest_model()
    if model is None:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"msg": "model not trained yet"})

    evaluation_data = dict()
    evaluation_data["model_id"] = ObjectId(model["_id"])
    evaluation_data["truth"] = dict()
    evaluation_data["is_prediction"] = True

    evaluation = await add_evaluation(evaluation_data)
    create_dir_if_not_exist(evaluation["evaluation_path"])
    create_dir_if_not_exist(os.path.join(evaluation["evaluation_path"], "data"))

    destination_file_path = os.path.join(evaluation["evaluation_path"], "data", file.filename)
    await write_to_file(file, destination_file_path)

    ml_model = model_store.get_running_model(model)
    if ml_model is None:
        return JSONResponse(status_code=status.HTTP_501_NOT_IMPLEMENTED, content={"msg": "model not trained yet"})

    pred = ml_model.predict(evaluation["evaluation_path"])

    result = dict()
    result[file.filename] = float(pred)
    evaluation["result"] = result
    return await update_evaluation(evaluation)


@router.get("/history", response_description="history of predictions", response_model=List[Evaluation])
async def evaluate_model(model_id: str = None):
    if model_id is None:
        evaluation_data_list = await fetch_all_history()
    else:
        evaluation_data_list = await fetch_history_for_model(model_id)

    return evaluation_data_list
