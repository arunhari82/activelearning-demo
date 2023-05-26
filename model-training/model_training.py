# %%
# install dependencies
%pip install -q -U pip
%pip install -q -r requirements.txt

# %%
# setup parameters
scratch = "../scratch/"

# %%
import numpy as np
import os, random, pathlib, warnings, itertools, math, json
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from PIL import ImageEnhance
from shutil import rmtree

import boto3
import botocore
from botocore import UNSIGNED
from botocore.client import Config

from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)

# %%
K.clear_session()

# %%
# setup paths for data
dataset = scratch + "Vegetable Images"

train_folder = os.path.join(dataset, "train")
test_folder = os.path.join(dataset, "validation")
validation_folder = os.path.join(dataset, "test")

# %%
IMAGE_SIZE = [224, 224]

inception = InceptionV3(
    input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False
)

for layer in inception.layers:
    layer.trainable = False

x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

prediction = Dense(15, activation="softmax")(x)

model = Model(inputs=inception.input, outputs=prediction)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

train_datagen = image.ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    train_folder, target_size=(224, 224), batch_size=64, class_mode="categorical"
)

test_set = test_datagen.flow_from_directory(
    test_folder, target_size=(224, 224), batch_size=64, class_mode="categorical"
)

class_map = training_set.class_indices
print(class_map)

# %%
model_metadata = "_inceptionV3_epoch5"
path_to_model = scratch + "model" + model_metadata + ".h5"

# %%
if not os.path.exists(path_to_model):
    r = model.fit(
        training_set,
        validation_data=test_set,
        epochs=5,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
    )
    
    model.save(path_to_model)

else:
    print("Model: already exists")


# %%
K.clear_session()

print("Model: Loading...")
model = load_model(path_to_model)
print("Model: Loaded")

# %%
category = {v: k for k, v in class_map.items()}
print(json.dumps(category, indent=4))


def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.0

    prediction = model.predict(img_processed)
    index = np.argmax(prediction)

    plt.title("Prediction - {}".format(category[index]))
    plt.imshow(img_array)

# %%
predict_image(os.path.join(validation_folder, "Cauliflower/1064.jpg"), model)

# %%
predict_image(os.path.join(validation_folder, "Bitter_Gourd/1202.jpg"), model)

# %%
predict_image(os.path.join(validation_folder, "Papaya/1266.jpg"), model)


