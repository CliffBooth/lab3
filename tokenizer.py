import tensorflow as tf
from app_const import *
import tensorflow.keras as keras
import pickle

def get_tokenizer(vocab=None):
    tokenizer = keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE,
        standardize=None,
        output_sequence_length=MAX_LENGTH,
        vocabulary=vocab,
    )
    return tokenizer