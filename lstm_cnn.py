"""
    The SC LSTM-CNN model for stock price prediction
"""
import os
import json
import functools
import argparse
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
    ENABLE_IMAGE_DECODE, PREDICT_SIZE, WINDOW_SIZE,
    VALIDATION_SP500_DATA_FILE, TEST_SP500_DATA_FILE, NUMBER_OF_VALIDATION
)
from util import get_cache_file, get_log_dir, set_gpu_memory_limit, RMAE
from cnn import create_residual_cnn_model
from lstm import create_lstm_model
from stock_dataset import load_lstm_cnn_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser('lstm', description='A LSTM model for stock price prediction')
    parser.add_argument("--name", help="The name of job", type=str, required=True)
    parser.add_argument("--epoch", help="Number of epoch", type=int, required=True)
    parser.add_argument("--batch_size", help="The size of training batch", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, required=True)
    parser.add_argument("--epsilon", help="Epsilon for learning", type=float, required=True)
    train_args = parser.parse_args()

    tf.print("---------------------------------------------")
    tf.print(train_args.name)
    tf.print(train_args.epoch)
    tf.print(train_args.batch_size)
    tf.print(train_args.learning_rate)
    tf.print(train_args.epsilon)
    tf.print("---------------------------------------------")

    set_gpu_memory_limit(0, 2000)
    if ENABLE_IMAGE_DECODE:
        train_dataset: tf.data.Dataset = load_lstm_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE,
            create_bar_filled_line_fusion_chart,
            TRAIN_SP500_DATA_FILE, 'train'
        )
        train_dataset = train_dataset.cache(get_cache_file('train', 'lstm_cnn'))
        train_dataset = train_dataset.batch(train_args.batch_size, drop_remainder=True)

        validation_dataset: tf.data.Dataset = load_lstm_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE,
            create_bar_filled_line_fusion_chart,
            VALIDATION_SP500_DATA_FILE, 'validation'
        )
        validation_dataset = validation_dataset.cache(get_cache_file('validation', 'lstm_cnn'))
        validation_dataset = validation_dataset.repeat().batch(train_args.batch_size, drop_remainder=True)

        test_dataset: tf.data.Dataset = load_lstm_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE,
            create_bar_filled_line_fusion_chart,
            TEST_SP500_DATA_FILE, 'test'
        )
        test_dataset = test_dataset.cache(get_cache_file('test', 'lstm_cnn'))
        test_dataset = test_dataset.batch(train_args.batch_size, drop_remainder=True)

        # Model
        chart_inputs = tf.keras.Input(shape=(112, 112, 3), batch_size=train_args.batch_size)
        log_input = tf.keras.Input(shape=(29, 2), batch_size=train_args.batch_size)
        _, conv_output = create_residual_cnn_model(chart_inputs)
        _, cell_output = create_lstm_model(log_input)

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
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_args.learning_rate, epsilon=train_args.epsilon
        )
        lstm_cnn_model.compile(
            optimizer=adam_optimizer, loss=tf.losses.MeanSquaredError(),
            metrics=['mape', tf.keras.metrics.RootMeanSquaredError(name='rmse'), RMAE]
        )
        validation_steps = int(NUMBER_OF_VALIDATION/train_args.batch_size)

        log_dir = get_log_dir("lstm_cnn", train_args.name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        tf.print("Training model")
        # Train model
        lstm_cnn_model.fit(
            train_dataset, epochs=train_args.epoch, verbose=1, batch_size=train_args.batch_size,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[tensorboard_callback]
        )

        tf.print("Evaluating LSTM-CNN model")
        result = lstm_cnn_model.evaluate(
            test_dataset, batch_size=train_args.batch_size,
            callbacks=[tensorboard_callback]
        )
        tf.print("Metric score:")
        tf.print(dict(zip(lstm_cnn_model.metrics_names, result)))
        with open(os.path.join(log_dir, "evaluation.json"), 'w') as evalfobj:
            json.dump(dict(zip(lstm_cnn_model.metrics_names, result)), evalfobj)
    else:
        logging.error("Please enable ENABLE_IMAGE_DECODE for training model")
