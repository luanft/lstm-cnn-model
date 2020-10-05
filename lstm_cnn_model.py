"""
    The SC LSTM-CNN model for stock price prediction
"""
import os
import functools
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from charts import (
    create_bar_candlestick_fusion_chart,
    create_bar_line_fusion_chart, create_bar_filled_line_fusion_chart
)
from setting import (
    TRAIN_SP500_DATA_FILE, ALPHA, BETA, GAMMA,
    ENABLE_IMAGE_DECODE, BATCH_SIZE,
    EPOCH, PREDICT_SIZE, WINDOW_SIZE
)
from shortcut_cnn_model import create_residual_cnn_model
from st_lstm import create_st_lstm
from stock_dataset import load_lstm_cnn_dataset


if __name__ == "__main__":
    if ENABLE_IMAGE_DECODE:
        train_dataset: tf.data.Dataset = load_lstm_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE,
            create_bar_filled_line_fusion_chart,
            TRAIN_SP500_DATA_FILE, 'train'
        )
        train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(100)

        # Model
        chart_inputs = tf.keras.Input(shape=(112, 112, 3), batch_size=BATCH_SIZE)
        log_input = tf.keras.Input(shape=(29, 2), batch_size=BATCH_SIZE)
        _, conv_output = create_residual_cnn_model(chart_inputs)
        _, cell_output = create_st_lstm(log_input)

        combined_feature = layers.Concatenate(axis=1)([cell_output, conv_output])
        flatten_layer = layers.Flatten()(combined_feature)

        full_connected_1 = layers.Dropout(0.5)(
            layers.Dense(500, activation='relu', use_bias=True)(flatten_layer)
        )
        full_connected_2 = layers.Dropout(0.5)(
            layers.Dense(100, activation='relu', use_bias=True)(full_connected_1)
        )
        full_connected_3 = layers.Dropout(0.5)(
            layers.Dense(25, activation='relu', use_bias=True)(full_connected_2)
        )
        lstm_cnn_output = layers.Dense(1, activation='linear')(full_connected_3)

        lstm_cnn_model = models.Model(
            inputs=[log_input, chart_inputs], outputs=lstm_cnn_output
        )
        lstm_cnn_model.summary()
        lstm_cnn_model.compile(
            optimizer='adam', loss=tf.losses.MeanSquaredError(),
            metrics=['mae', 'mse', 'mape']
        )
        # Train model
        lstm_cnn_model.fit(train_dataset, epochs=EPOCH, verbose=1, batch_size=BATCH_SIZE)
    else:
        logging.error("Please enable ENABLE_IMAGE_DECODE for training model")
