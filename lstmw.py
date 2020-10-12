"""
    Shortcut connection CNN model
"""
import os
import json
import argparse
import logging
import tensorflow as tf

from tensorflow_addons.optimizers import AdamW
from tensorflow.keras import layers, models
from setting import (ENABLE_IMAGE_DECODE, PREDICT_SIZE,
                     TEST_SP500_DATA_FILE, TRAIN_SP500_DATA_FILE,
                     VALIDATION_SP500_DATA_FILE, WINDOW_SIZE, NUMBER_OF_VALIDATION)
from stock_dataset import load_lstm_dataset
from util import get_cache_file, get_log_dir, set_gpu_memory_limit, RMAE


def create_lstm_model(log_inputs: tf.keras.Input) -> layers.Layer:
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
    parser = argparse.ArgumentParser('lstm', description='A LSTM model for stock price prediction')
    parser.add_argument("--name", help="The name of job", type=str, required=True)
    parser.add_argument("--epoch", help="Number of epoch", type=int, required=True)
    parser.add_argument("--batch_size", help="The size of training batch", type=int, required=True)
    parser.add_argument("--learning_rate", help="Learning rate", type=float, required=True)
    parser.add_argument("--epsilon", help="Epsilon for learning", type=float, required=True)
    parser.add_argument("--weight_decay", help="Epsilon for learning", type=float, required=True)
    train_args = parser.parse_args()

    tf.print("---------------------------------------------")
    tf.print(train_args.name)
    tf.print(train_args.epoch)
    tf.print(train_args.batch_size)
    tf.print(train_args.learning_rate)
    tf.print(train_args.epsilon)
    tf.print(train_args.weight_decay)
    tf.print("---------------------------------------------")


    set_gpu_memory_limit(0, 2000)
    if ENABLE_IMAGE_DECODE:
        # Dataset
        train_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, TRAIN_SP500_DATA_FILE
        )
        train_dataset = train_dataset.cache(get_cache_file("train", "lstm"))
        train_dataset = train_dataset.batch(train_args.batch_size, drop_remainder=True)

        validation_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, VALIDATION_SP500_DATA_FILE
        )
        validation_dataset = validation_dataset.cache(get_cache_file("validation", "lstm"))
        validation_dataset = validation_dataset.repeat().batch(train_args.batch_size, drop_remainder=True)

        test_dataset: tf.data.Dataset = load_lstm_dataset(
            WINDOW_SIZE, PREDICT_SIZE, TEST_SP500_DATA_FILE
        )
        test_dataset = test_dataset.cache(get_cache_file("test", "lstm"))
        test_dataset = test_dataset.batch(train_args.batch_size, drop_remainder=True)


        # LSTM model
        log_inputs = tf.keras.Input(shape=(29, 2), batch_size=train_args.batch_size)
        st_lstm_model = models.Model(inputs=log_inputs, outputs=create_lstm_model(log_inputs)[0])
        st_lstm_model.summary()
        log_dir = get_log_dir("lstm_adadelta", train_args.name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        adamw_optimizer = AdamW(
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay, epsilon=train_args.epsilon
        )
        st_lstm_model.compile(
            optimizer=adamw_optimizer, loss=tf.losses.MeanSquaredError(),
            metrics=['mape', tf.keras.metrics.RootMeanSquaredError(name='rmse'), RMAE]
        )
        validation_steps = int(NUMBER_OF_VALIDATION / train_args.batch_size)
        # training model
        tf.print("Training model")
        st_lstm_model.fit(
            train_dataset, epochs=train_args.epoch, verbose=1, batch_size=train_args.batch_size,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[tensorboard_callback]
        )
        tf.print("Evaluating model")
        result = st_lstm_model.evaluate(
            test_dataset, batch_size=train_args.batch_size,
            callbacks=[tensorboard_callback]
        )
        tf.print("Metric score:")
        tf.print(dict(zip(st_lstm_model.metrics_names, result)))
        with open(os.path.join(log_dir, "evaluation.json"), 'w') as evalfobj:
            json.dump(dict(zip(st_lstm_model.metrics_names, result)), evalfobj)
    else:
        logging.error("Please enable ENABLE_IMAGE_DECODE for training model")


