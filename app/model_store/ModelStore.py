from app.ml_models.mobileNetModel import MobileNetBasedModel
import os

class ModelStore:
    running_model = dict()

    def get_running_model(self, model, version):
        if model["_id"] not in self.running_model:
            print("starting model")
            self.running_model[model["_id"]] = MobileNetBasedModel(os.path.join(model["training_dataset_path"], version),
                                                                   os.path.join(model["model_path"], version))
            self.running_model[model["_id"]].start()
        return self.running_model[model["_id"]]

model_store = ModelStore()