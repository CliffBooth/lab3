import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing import image
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from app_data import captions
from app_model import get_model, tokenizer
from app_const import *

model_path = os.path.join("models", f"{datetime.now().strftime('%d.%m_%H-%M')}")

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(captions['image'], captions['caption']):
    img_to_cap_vector[img].append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index],
                                          img_keys[slice_index:])

train_imgs = []
train_captions = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    train_imgs.extend([imgt] * capt_len)
    train_captions.extend(img_to_cap_vector[imgt])

val_imgs = []
val_captions = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector[imgv])

def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = img / 255.
    caption = tokenizer(caption)
    return img, caption

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_captions))

train_dataset = train_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

caption_model = get_model()

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
tb_callback = TensorBoard(f'{model_path}/logs', update_freq=1)
weights_checkpoint = ModelCheckpoint(f"{model_path}/pretrained_weights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)

history = caption_model.fit(
    train_dataset,
    epochs=1,
    validation_data=val_dataset,
    callbacks=[
        early_stopping,
        tb_callback,
        weights_checkpoint
    ]
)

save_model_path = f"{model_path}/model"
# caption_model.save(save_model_path, save_format="tf")
tf.saved_model.save(caption_model, save_model_path)

caption_model.cnn_model.save_weights(f'{model_path}/cnn_model')
caption_model.encoder.save_weights(f'{model_path}/encoder')
caption_model.decoder.save_weights(f'{model_path}/decoder')