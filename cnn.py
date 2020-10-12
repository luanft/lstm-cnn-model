"""
    Shortcut connection CNN model
"""
import os
import argparse
import json
import logging
import tensorflow as tf
from tensorflow.keras import layers, models

from charts import create_bar_filled_line_fusion_chart
from setting import (ENABLE_IMAGE_DECODE, PREDICT_SIZE,
                     TEST_SP500_DATA_FILE, TRAIN_SP500_DATA_FILE,
                     VALIDATION_SP500_DATA_FILE, WINDOW_SIZE,
                     NUMBER_OF_VALIDATION)

from stock_dataset import load_cnn_dataset

from util import get_cache_file, get_log_dir, set_gpu_memory_limit, RMAE


def create_res_conv1_layer(input):
    """Create the res conv1 layer"""
    shortcut_conv1 = layers.Conv2D(
        128, (1, 1), strides=1, padding='VAlID',
        input_shape=(28, 28, 32), name="shortcut_conv1"
    )(input)

    resconv1_1x1_kernel_in = layers.Conv2D(
        128, (1, 1), strides=1, padding='VAlID',
        input_shape=(28, 28, 32), name="resconv1_1x1_kernel_in", activation='relu'
    )(input)

    resconv1_3x3_kernel = layers.Conv2D(
        128, (3, 3), strides=1, padding='SAME',
        input_shape=(28, 28, 128), name="resconv1_3x3_kernel", activation='relu'
    )(resconv1_1x1_kernel_in)

    conv1x1_out = layers.Conv2D(
        128, (1, 1), strides=1, padding='VALID',
        input_shape=(28, 28, 128), name="resconv1_1x1_kernel_out", activation='relu'
    )(resconv1_3x3_kernel)
    return layers.ReLU()(layers.Add(name="res_conv1_add_shortcut1")([conv1x1_out, shortcut_conv1]))


def create_res_conv2_layer(input):
    """Create the res conv1 layer"""
    shortcut_conv2 = layers.Conv2D(
        256, (1, 1), strides=2, padding='VAlID',
        input_shape=(28, 28, 128), name="shortcut_conv2"
    )(input)

    resconv2_1x1_kernel_in = layers.Conv2D(
        256, (1, 1), strides=2, padding='VAlID',
        input_shape=(28, 28, 128), name="resconv2_1x1_kernel_in", activation='relu'
    )(input)

    resconv2_3x3_kernel = layers.Conv2D(
        128, (3, 3), strides=1, padding='SAME',
        input_shape=(14, 14, 256), name="resconv2_3x3_kernel", activation='relu'
    )(resconv2_1x1_kernel_in)

    resconv2_1x1_kernel_out = layers.Conv2D(
        256, (1, 1), strides=1, padding='VALID',
        input_shape=(14, 14, 256), name="resconv2_1x1_kernel_out", activation='relu'
    )(resconv2_3x3_kernel)
    return layers.ReLU()(layers.Add(name="res_conv2_add_shortcut2")([resconv2_1x1_kernel_out, shortcut_conv2]))


def create_res_conv3_layer(input):
    """Create the res conv1 layer"""
    shortcut_conv3 = layers.Conv2D(
        512, (1, 1), strides=2, padding='VAlID',
        input_shape=(28, 28, 256), name="shortcut_conv3"
    )(input)

    resconv3_1x1_kernel_in = layers.Conv2D(
        512, (1, 1), strides=2, padding='VAlID',
        input_shape=(28, 28, 256), name="resconv3_1x1_kernel_in", activation='relu'
    )(input)

    resconv3_3x3_kernel = layers.Conv2D(
        512, (3, 3), strides=1, padding='SAME',
        input_shape=(7, 7, 512), name="resconv3_3x3_kernel", activation='relu'
    )(resconv3_1x1_kernel_in)

    resconv3_1x1_kernel_out = layers.Conv2D(
        512, (1, 1), strides=1, padding='VALID',
        input_shape=(7, 7, 512), name="resconv3_1x1_kernel_out", activation='relu'
    )(resconv3_3x3_kernel)
    return layers.ReLU()(layers.Add(name="res_conv3_add_shortcut3")([resconv3_1x1_kernel_out, shortcut_conv3]))


def zero_pad2d(x, padding):
    """Zero padding 2d for inputs"""
    return layers.ZeroPadding2D(padding=padding, data_format="channels_last")(x)


def create_residual_cnn_model(chart_inputs: tf.keras.Input) -> models.Model:
    """Create the residual leaning CNN model"""
    # Conv1 layer
    conv1_layer = layers.Conv2D(
        32, (7, 7), strides=(2, 2), padding='valid',
        input_shape=(112, 112, 3), name="CONV1", activation="relu",
        data_format="channels_last"
    )(zero_pad2d(chart_inputs, 3))

    # Max pooling layer
    max_pool_layer = layers.MaxPool2D(
        pool_size=(3, 3), padding='VALID', strides=(2, 2),
        name="MaxPooling"
    )(conv1_layer)

    res_conv1 = create_res_conv1_layer(max_pool_layer)
    res_conv2 = create_res_conv2_layer(res_conv1)
    res_conv3 = create_res_conv3_layer(res_conv2)
    average_pool = layers.Flatten()(layers.AveragePooling2D(pool_size=(7,7))(res_conv3))

    full_connected_1 = layers.Dropout(0.5)(
        layers.Dense(500, activation='relu', use_bias=True)(average_pool)
    )
    full_connected_2 = layers.Dropout(0.5)(
        layers.Dense(100, activation='relu', use_bias=True)(full_connected_1)
    )
    full_connected_3 = layers.Dropout(0.5)(
        layers.Dense(25, activation='relu', use_bias=True)(full_connected_2)
    )
    # return full_connected_3
    output_layer = layers.Dense(1, activation='linear')(full_connected_3)
    return output_layer, average_pool


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
        train_dataset: tf.data.Dataset = load_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE, create_bar_filled_line_fusion_chart,
            path_to_csv=TRAIN_SP500_DATA_FILE, dataset_type='train'
        )
        train_dataset = train_dataset.cache(get_cache_file('train', 'cnn'))
        train_dataset = train_dataset.batch(train_args.batch_size, drop_remainder=True)


        validation_dataset: tf.data.Dataset = load_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE, create_bar_filled_line_fusion_chart,
            path_to_csv=VALIDATION_SP500_DATA_FILE, dataset_type='validation'
        )
        validation_dataset = validation_dataset.cache(get_cache_file('validation', 'cnn'))
        validation_dataset = validation_dataset.repeat().batch(train_args.batch_size, drop_remainder=True)


        test_dataset: tf.data.Dataset = load_cnn_dataset(
            WINDOW_SIZE, PREDICT_SIZE, create_bar_filled_line_fusion_chart,
            path_to_csv=TEST_SP500_DATA_FILE, dataset_type='test'
        )
        test_dataset = test_dataset.cache(get_cache_file('test', 'cnn'))
        test_dataset = test_dataset.batch(train_args.batch_size, drop_remainder=True)

        # Model
        chart_inputs = tf.keras.Input(batch_input_shape=(train_args.batch_size, 112, 112, 3))
        cnn_model = models.Model(
            inputs=chart_inputs, 
            outputs=create_residual_cnn_model(chart_inputs)[0]
        )
        cnn_model.summary()

        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=train_args.learning_rate, epsilon=train_args.epsilon
        )

        log_dir = get_log_dir("cnn", train_args.name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        cnn_model.compile(
            optimizer=adam_optimizer, loss=tf.losses.MeanSquaredError(),
            metrics=['mape', tf.keras.metrics.RootMeanSquaredError(name='rmse'), RMAE]
        )
        validation_steps = int(NUMBER_OF_VALIDATION/train_args.batch_size)

        tf.print("Training model")
        # training model
        cnn_model.fit(
            train_dataset, epochs=train_args.epoch, verbose=1, batch_size=train_args.batch_size,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[tensorboard_callback]
        )
        tf.print("Evaluating CNN model")
        result = cnn_model.evaluate(
            test_dataset, batch_size=train_args.batch_size,
            callbacks=[tensorboard_callback]
        )
        tf.print("Metric score:")
        tf.print(dict(zip(cnn_model.metrics_names, result)))
        with open(os.path.join(log_dir, "evaluation.json"), 'w') as evalfobj:
            json.dump(dict(zip(cnn_model.metrics_names, result)), evalfobj)
    else:
        logging.error("Please enable ENABLE_IMAGE_DECODE for training model")
