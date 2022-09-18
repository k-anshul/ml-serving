from ml_models.mobileNetModel import MobileNetBasedModel

class ModelStore:
    running_model = dict()

    def load_model(self, model_data):
        self.running_model[model_data["id"]] = MobileNetBasedModel(model_data["training_dataset_path"], model_data["model_path"])

    def get_running_model(self, model):
        if model["id"] not in self.running_model:
            print("starting model")
            self.running_model[model["id"]] = MobileNetBasedModel(model["training_dataset_path"], model["model_path"])
        return self.running_model[model["id"]]

model_store = ModelStore()