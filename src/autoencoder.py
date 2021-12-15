import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import process_data
import validate_data


class AutoEncoder:
    def __init__(self) -> None:
        pass

    def load_data(self):
        dataloader = process_data.DataPrepare()
        train_data, test_data = dataloader.process_data()
        data_dev = validate_data.DatavalidationTest(train_data)
        x_train, x_test, y_train, y_test = data_dev.dataset_development()
        return x_train, x_test, y_train, y_test

    def model(self):
        x_train, x_test, y_train, y_test = self.load_data()
        input_dim = x_train.shape[1]
        encoding_dim = 14
        input_layer = Input(shape=(input_dim,))

        encoder = Dense(
            encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5)
        )(input_layer)
        encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

        decoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
        decoder = Dense(input_dim, activation="relu")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)

        nb_epoch = 100
        batch_size = 32

        autoencoder.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
        )

        checkpointer = ModelCheckpoint(
            filepath="model.h5", verbose=0, save_best_only=True
        )
        tensorboard = TensorBoard(
            log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True
        )

        history = autoencoder.fit(
            x_train,
            y_train,
            epochs=nb_epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test),
            verbose=1,
            callbacks=[checkpointer, tensorboard],
        ).history
        return autoencoder


dataloader = process_data.DataPrepare()
train_data, test_data = dataloader.process_data()
data_dev = validate_data.DatavalidationTest(train_data)
x_train, x_test, y_train, y_test = data_dev.dataset_development()

dl_model = AutoEncoder()
# autoencoder = dl_model.model() 
autoencoder = load_model("model.h5") 
predictions = autoencoder.predict(x_test) 

# load model.h5


# get the prediction in 0 or 1 format

# print(predictions)

threshold = 0.9
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if predictions[i][j] > threshold:
            predictions[i][j] = 1
        else:
            predictions[i][j] = 0
print(np.unique(predictions))

