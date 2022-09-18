from ml_models.mobileNetModel import MobileNetBasedModel

class ModelStore:
    running_model = dict()

    def load_model(self, model):
        self.running_model[model["_id"]] = MobileNetBasedModel(model["training_dataset_path"], model["model_path"])

    def get_running_model(self, model):
        if model["_id"] not in self.running_model:
            print("starting model")
            self.running_model[model["_id"]] = MobileNetBasedModel(model["training_dataset_path"], model["model_path"])
        return self.running_model[model["_id"]]

model_store = ModelStore()