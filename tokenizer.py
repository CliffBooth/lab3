from const import *
import tensorflow.keras as keras

def get_tokenizer(vocab=None):
    tokenizer = keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE,
        standardize=None,
        output_sequence_length=MAX_LENGTH,
        # vocabulary=vocab,
    )
    return tokenizer