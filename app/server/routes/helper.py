from bson import ObjectId
from fastapi import HTTPException
from ..database import (
    get_model_by_id,
    update_status
)
from app.ml_models.mobileNetModel import MobileNetBasedModel
from fastapi.concurrency import run_in_threadpool
from ..constants import TrainingStatus

import os


# todo :: write tests
async def get_model(model_id, version):
    try:
        object_id = ObjectId(model_id)
    except Exception as e:
        # id is not valid mongo object id
        raise HTTPException(status_code=400, detail=e.args)

    model = await get_model_by_id(object_id, version)
    if model is None:
        # model with id and version not found
        print("model not found")
        raise HTTPException(status_code=404, detail=f"Model {0} with version {1} not found".format(model_id, version))
    return model


async def train_model(model, version):
    ml_model = MobileNetBasedModel(os.path.join(model["training_dataset_path"], version),
                                   os.path.join(model["model_path"], version))
    try:
        # train model
        await run_in_threadpool(lambda: ml_model.train())
        # on success mark model as trained
        await update_status(model["_id"], version, TrainingStatus.TRAINED)
    except Exception as e:
        print(e)
        # on exception mark model as failure
        await update_status(model["_id"], version, TrainingStatus.FAILED)