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
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

img_path = 'flickr8k/images/10815824_2997e03d76.jpg'
img = image.load_img(img_path, target_size=(299, 299))

x = image.img_to_array(img)  # Convert the image to an array
x = np.expand_dims(x, axis=0)  # Add an extra dimension for the batch
x = preprocess_input(x)  # Preprocess the input image

preprocessed_img = preprocess_input(x)

model = InceptionV3(weights='imagenet')

predictions = model.predict(x)
decoded_predictions = decode_predictions(predictions, top=3)[0]

for pred in decoded_predictions:
    # print(f'{pred[1]}: {pred[2]*100:.2f}%')
    print(pred)