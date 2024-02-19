import re
import tensorflow as tf
import collections
import random

import pandas as pd

from const import *
from tokenizer import get_tokenizer

tokenizer = get_tokenizer()

captions = pd.read_csv('flickr8k/captions.txt')
captions['image'] = captions['image'].apply(lambda x: f'flickr8k/images/{x}')

tokenizer.adapt(captions['caption'])

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

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text

captions['caption'] = captions['caption'].apply(preprocess)

def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = img / 255.
    caption = tokenizer(caption)
    return img, caption

def get_datasets():
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

    return train_dataset, val_dataset

def get_test_dataset():
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_imgs[:10], train_captions[:10])
    )
    train_dataset = train_dataset.map(
        load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset