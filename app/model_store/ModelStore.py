from app.ml_models.mobileNetModel import MobileNetBasedModel
import os


class ModelStore:
    # cache of running models
    # todo :: use LRU cache
    running_model = dict()

    def get_running_model(self, model, version):
        load_model = MobileNetBasedModel(os.path.join(model["training_dataset_path"], version),
                                         os.path.join(model["model_path"], version))
        load_model.start()
        return load_model

    def get_running_cached_model(self, model, version):
        if model["_id"] not in self.running_model:
            print("starting model")
            self.running_model[model["_id"]] = MobileNetBasedModel(
                os.path.join(model["training_dataset_path"], version),
                os.path.join(model["model_path"], version))
            self.running_model[model["_id"]].start()
        return self.running_model[model["_id"]]


model_store = ModelStore()
