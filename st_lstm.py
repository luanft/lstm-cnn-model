"""
    Shortcut connection CNN model
"""
import logging

import tensorflow as tf
from tensorflow.keras import layers, models

from setting import (BATCH_SIZE, ENABLE_IMAGE_DECODE, EPOCH, PREDICT_SIZE,
                     WINDOW_SIZE, TRAIN_SP500_DATA_FILE, TEST_SP500_DATA_FILE, VALIDATION_SP500_DATA_FILE)
from stock_dataset import load_lstm_dataset


def create_st_lstm(log_inputs: tf.keras.Input) -> layers.Layer:
    lstm_layer_1 = layers.LSTM(29, batch_input_shape=(None, 29, 2), return_sequences=True)(log_inputs)
    lstm_layer_2 = layers.LSTM(29, batch_input_shape=(None, 29, 29))(lstm_layer_1)
    flatten_layer = layers.Flatten()(lstm_layer_2)
    full_connected_1 = layers.Dropout(0.5)(
        layers.Dense(500, activation='relu', use_bias=True)(flatten_layer)
    )
    full_connected_2 = layers.Dropout(0.5)(
        layers.Dense(100, activation='relu', use_bias=True)(full_connected_1)
    )
    full_connected_3 = layers.Dropout(0.5)(
        layers.Dense(25, activation='relu', use_bias=True)(full_connected_2)
    )
    output_layer = layers.Dense(1, activation='linear')(full_connected_3)
    return output_layer, flatten_layer

if __name__ == "__main__":
    if ENABLE_IMAGE_DECODE:
        train_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, TRAIN_SP500_DATA_FILE
        )
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(100)

        validation_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, VALIDATION_SP500_DATA_FILE
        )
        validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(100).repeat()

        test_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, TEST_SP500_DATA_FILE
        )
        test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(100)

        # Model
        log_inputs = tf.keras.Input(shape=(29, 2), batch_size=BATCH_SIZE)
        st_lstm_model = models.Model(inputs=log_inputs, outputs=create_st_lstm(log_inputs)[0])
        st_lstm_model.summary()

        st_lstm_model.compile(
            optimizer='adam', loss=tf.losses.MeanSquaredError(),
            metrics=['mae', 'mse', 'mape']
        )
        # training model
        st_lstm_model.fit(
            train_dataset, epochs=EPOCH, verbose=1, batch_size=BATCH_SIZE,
            validation_data=validation_dataset, validation_batch_size=BATCH_SIZE,
            validation_steps=10
        )
        st_lstm_model.evaluate(test_dataset, batch_size=BATCH_SIZE)
    else:
        logging.error("Please enable ENABLE_IMAGE_DECODE for training model")
