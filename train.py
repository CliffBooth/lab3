import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from data import get_datasets
from model import get_model

model_path = os.path.join("models", f"{datetime.now().strftime('%d.%m_%H-%M')}")

train_dataset, val_dataset = get_datasets()

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
tf.saved_model.save(caption_model, save_model_path)

caption_model.cnn_model.save_weights(f'{model_path}/cnn_model')
caption_model.encoder.save_weights(f'{model_path}/encoder')
caption_model.decoder.save_weights(f'{model_path}/decoder')