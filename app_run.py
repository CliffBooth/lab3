import os.path
from random import random

from app_model import get_model, tokenizer
from matplotlib import pyplot as plt
import tensorflow as tf
from app_const import *
import numpy as np
import keras

weights_path = f"models/15.02_15-06/pretrained_weights.h5"
saved_model_path = 'models/15.02_18-48/model'

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = img / 255.
    return img

word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

def generate_caption(caption_model, img_path):
    img = load_image_from_path(img_path)
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)

        pred_idx = np.argmax(pred[0, i, :])
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break

        y_inp += ' ' + pred_word

    y_inp = y_inp.replace('[start] ', '')
    return y_inp


# model = keras.models.load_model(saved_model_path)
# model = tf.saved_model.load(saved_model_path)

model = get_model()
# model.build(input_shape = (1, 299, 299, 3))
# model(np.zeros((1, 299, 299, 3)))
# dummy_input = tf.ones((1, 299, 299, 3))
test_prediction = generate_caption(model, 'flickr8k/images/3006093003_c211737232.jpg')
print("test_prediction = ", test_prediction)
model.load_weights(weights_path)

# test...

what_is_it = idx2word(2).numpy().decode('utf-8')
print(what_is_it)

pic_path = 'flickr8k/images/3006093003_c211737232.jpg'
pred_caption = generate_caption(model, pic_path)
print('image path: ', pic_path)
print('Predicted Caption:', pred_caption)
plt.imshow(pic_path)
plt.show()

# idx = random.randrange(0, len(val_imgs))
# img_path = val_imgs[idx]
#
