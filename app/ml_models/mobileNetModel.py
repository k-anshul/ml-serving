import tensorflow as tf
import os
import shutil

from glob import glob
from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

avg_sizes = (397, 364, 3)
avg_size = (397, 364)

class MobileNetBasedModel:

    def __init__(self, input_path, model_output):
        self.input_path = input_path
        self.model_output = model_output

    def predict(self, test_images_path):
        print(test_images_path)
        model = tf.keras.models.load_model(self.model_output)
        test_set = self.test_generator(batch_size=64, path_to_test=test_images_path)
        preds = model.predict(test_set).flatten()
        return preds

    def test_generator(self, batch_size, path_to_test):

        testing_preprocessor = ImageDataGenerator(
            rescale=1 / 255.
        )

        test_generator = testing_preprocessor.flow_from_directory(
            path_to_test,
            class_mode="binary",
            target_size=avg_size,
            color_mode="rgb",
            shuffle=False,
            batch_size=batch_size
        )

        return test_generator

    def mobileNet_based_model(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=avg_sizes, include_top=False, weights='imagenet')
        base_model.trainable = False
        input_layer = Input(shape=avg_sizes),
        x = base_model(input_layer, training=False)
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=64, activation="relu")(x)
        x = Dense(units=32, activation="relu")(x)
        x = Dense(units=1, activation="sigmoid")(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def split_data(self, split_size=0.2):
        dirs = os.listdir(self.input_path)
        for dir in dirs:
            dir_path = os.path.join(self.input_path, dir)
            images_paths = glob(os.path.join(dir_path, "*.jpg"))
            if len(images_paths) == 0 :
                continue
            # print(images_paths)
            train_set, val_set = train_test_split(images_paths, test_size=split_size)
            path_to_validation = os.path.join(self.input_path, "val\\" + dir)
            if not os.path.isdir(path_to_validation):
                os.makedirs(path_to_validation)

            path_to_training = os.path.join(self.input_path, "train\\" + dir)

            if not os.path.isdir(path_to_training):
                os.makedirs(path_to_training)

            for x in train_set:
                # print(x)
                shutil.copy(x, path_to_training)

            for x in val_set:
                # print(x)
                shutil.copy(x, path_to_validation)
            shutil.rmtree(dir_path)

    def train_val_generators(self, batch_size):
        path_to_train = os.path.join(self.input_path, "train")
        path_to_val = os.path.join(self.input_path, "val")

        training_preprocessor = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_preprocessor = ImageDataGenerator(
            rescale=1 / 255.
        )

        train_generator = training_preprocessor.flow_from_directory(
            path_to_train,
            class_mode="binary",
            target_size=avg_size,
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size
        )

        val_generator = validation_preprocessor.flow_from_directory(
            path_to_val,
            class_mode="binary",
            target_size=avg_size,
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size
        )

        return train_generator, val_generator

    def train(self):
        self.mobileNet_based_model()
        self.split_data(0.3)
        batch_size = 64
        train, val = self.train_val_generators(batch_size=batch_size)
        epochs = 8

        ckpt_saver = ModelCheckpoint(
            filepath=self.model_output,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch",
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model.fit(
            train,
            validation_data=val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop, ckpt_saver]
        )