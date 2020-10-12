"""
    The SC LSTM-CNN model for stock price prediction
"""
import os
import functools
import multiprocessing
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras import layers, models
from charts import (
    create_bar_candlestick_fusion_chart,
    create_bar_line_fusion_chart,
    create_bar_filled_line_fusion_chart
)
from setting import (
    TRAIN_SP500_DATA_FILE, ALPHA, BETA, GAMMA, ENABLE_IMAGE_DECODE,
    WINDOW_SIZE, BATCH_SIZE, EPOCH, PREDICT_SIZE, TEST_SP500_DATA_FILE, 
    VALIDATION_SP500_DATA_FILE, CHART_DIR
)
from util import convert_to_list

def load_csv_data(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path, batch_size=1, na_value="0",
        header=True, shuffle=False, num_epochs=1,
        ignore_errors=True, **kwargs
    )
    return dataset


def make_dataset_generator_for_lstmcnn_model(fn_draw_chart, window_size=30, predict_size=5, path_to_csv='', dataset_type=''):
    chart_dir = os.path.join(CHART_DIR, dataset_type)
    def create_window_timeseries_dataset():
        """window + predict_size time series dataset"""
        frame_index = 0
        dataset = load_csv_data(path_to_csv)
        sidding_window_dataset: tf.data.Dataset = dataset.window(
            size=window_size + predict_size, shift=1,
            drop_remainder=True
        )
        data_frame: tf.data.Dataset

        for data_frame in sidding_window_dataset.take(-1):
            all_frame_close_prices = convert_to_list(data_frame['Trade Close'])
            predict_price = all_frame_close_prices[window_size + predict_size - 1]
            data_frame['Time'] = data_frame['Time'].take(window_size)
            data_frame['Trade High'] = data_frame['Trade High'].take(window_size)
            data_frame['Trade Low'] = data_frame['Trade Low'].take(window_size)
            data_frame['Trade Volume'] = data_frame['Trade Volume'].take(window_size)
            data_frame['Trade Open'] = data_frame['Trade Open'].take(window_size)
            data_frame['Trade Close'] = data_frame['Trade Close'].take(window_size)
            trade_volume = convert_to_list(data_frame['Trade Volume'])
            close_price = convert_to_list(data_frame['Trade Close'])
            normed_close_price = np.log10(np.array(close_price[:-1])) /  np.log10(np.array(close_price[1:]))
            normed_trade_volume = np.log10(np.array(trade_volume[:-1])) /  np.log10(np.array(trade_volume[1:]))
            frame_index += 1
            yield (
                tf.convert_to_tensor(np.reshape(np.array([normed_close_price, normed_trade_volume]), [29, 2])), 
                fn_draw_chart("frame_%s" % frame_index, data_frame, chart_dir=chart_dir)
            ), predict_price
    return create_window_timeseries_dataset


def make_dataset_generator_for_lstm(window_size=30, predict_size=5, path_to_csv=''):
    def create_window_timeseries_dataset():
        """window + predict_size time series dataset"""
        dataset = load_csv_data(path_to_csv)
        sidding_window_dataset: tf.data.Dataset = dataset.window(
            size=window_size + predict_size, shift=1,
            drop_remainder=True
        )
        data_frame: tf.data.Dataset

        for data_frame in sidding_window_dataset.take(-1):
            all_frame_close_prices = convert_to_list(data_frame['Trade Close'])
            predict_price = all_frame_close_prices[window_size + predict_size - 1]
            data_frame['Trade Volume'] = data_frame['Trade Volume'].take(window_size)
            data_frame['Trade Close'] = data_frame['Trade Close'].take(window_size)
            trade_volume = convert_to_list(data_frame['Trade Volume'])
            close_price = convert_to_list(data_frame['Trade Close'])
            normed_close_price = np.log10(np.array(close_price[:-1])) /  np.log10(np.array(close_price[1:]))
            normed_trade_volume = np.log10(np.array(trade_volume[:-1])) /  np.log10(np.array(trade_volume[1:]))
            log_input = tf.convert_to_tensor(
                np.reshape(np.array([normed_close_price, normed_trade_volume]), [29, 2])
            )
            yield log_input, predict_price
    return create_window_timeseries_dataset


def make_dataset_generator_for_chart_data(fn_draw_chart, window_size=30, predict_size=5, path_to_csv='', dataset_type=''):
    chart_dir = os.path.join(CHART_DIR, dataset_type)
    def create_window_timeseries_dataset():
        """window + predict_size time series dataset"""
        frame_index = 0
        dataset = load_csv_data(path_to_csv)
        sidding_window_dataset: tf.data.Dataset = dataset.window(
            size=window_size + predict_size, shift=1,
            drop_remainder=True
        )
        data_frame: tf.data.Dataset
        for data_frame in sidding_window_dataset.take(-1):
            all_frame_close_prices = convert_to_list(data_frame['Trade Close'])
            predict_price = all_frame_close_prices[window_size + predict_size - 1]
            data_frame['Time'] = data_frame['Time'].take(window_size)
            data_frame['Trade High'] = data_frame['Trade High'].take(window_size)
            data_frame['Trade Low'] = data_frame['Trade Low'].take(window_size)
            data_frame['Trade Volume'] = data_frame['Trade Volume'].take(window_size)
            data_frame['Trade Open'] = data_frame['Trade Open'].take(window_size)
            data_frame['Trade Close'] = data_frame['Trade Close'].take(window_size)
            frame_index += 1
            yield fn_draw_chart("frame_%s" % frame_index, data_frame, chart_dir=chart_dir), predict_price
    return create_window_timeseries_dataset


def fn_task_generate_filled_line_fusion(path_to_csv, dataset_type):
    """Process data"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_chart_data(
            create_bar_filled_line_fusion_chart,
            window_size=WINDOW_SIZE,
            predict_size=PREDICT_SIZE,
            path_to_csv=path_to_csv,
            dataset_type=dataset_type
        ),
        output_types=(tf.float64, tf.float16),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
    )
    total = 0
    for _record in dataset.batch(1).take(-1):
        logging.warning(total)
        total += 1


def fn_task_generate_candlestick_fusion(path_to_csv, dataset_type):
    """Process data"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_chart_data(
            create_bar_candlestick_fusion_chart,
            window_size=WINDOW_SIZE,
            predict_size=PREDICT_SIZE,
            path_to_csv=path_to_csv,
            dataset_type=dataset_type
        ),
        output_types=(tf.float64, tf.float16),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
    )
    total = 0
    for _record in dataset.batch(1).take(-1):
        logging.warning(total)
        total += 1


def fn_task_generate_barline_fusion(path_to_csv, dataset_type):
    """Process data"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_chart_data(
            create_bar_line_fusion_chart,
            window_size=WINDOW_SIZE,
            predict_size=PREDICT_SIZE,
            path_to_csv=path_to_csv,
            dataset_type=dataset_type
        ),
        output_types=(tf.float64, tf.float16),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([]))
    )
    total = 0
    for _record in dataset.batch(1).take(-1):
        logging.warning(total)
        total += 1


def generate_chart_dataset(path_to_csv, dataset_type):
    """create the dataset"""
    worker1 = multiprocessing.Process(
        target=fn_task_generate_filled_line_fusion, args=(path_to_csv, dataset_type)
    )
    worker2 = multiprocessing.Process(
        target=fn_task_generate_candlestick_fusion, args=(path_to_csv, dataset_type)
    )
    worker3 = multiprocessing.Process(
        target=fn_task_generate_barline_fusion, args=(path_to_csv, dataset_type)
    )
    worker1.start()
    worker2.start()
    worker3.start()
    worker1.join()
    worker2.join()
    worker3.join()


def load_lstm_dataset(window_size, predict_size, path_to_csv):
    """Load lstm dataset"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_lstm(
            window_size=window_size, predict_size=predict_size, path_to_csv=path_to_csv
        ),
        output_types=(tf.float64, tf.float16),
        output_shapes=(tf.TensorShape((29, 2)), tf.TensorShape([]))
    )
    return dataset


def load_cnn_dataset(window_size, predict_size, fn_draw_chart, path_to_csv, dataset_type):
    """Load CNN dataset"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_chart_data(
            fn_draw_chart, window_size=window_size, predict_size=predict_size,
            path_to_csv=path_to_csv, dataset_type=dataset_type
        ),
        output_types=(tf.float64, tf.float16),
        output_shapes=(tf.TensorShape((112, 112, 3)), tf.TensorShape([]))
    )
    return dataset
    # return dataset.take(-1)


def load_lstm_cnn_dataset(window_size, predict_size, fn_draw_chart, path_to_csv, dataset_type):
    """Load LSTM CNN dataset"""
    dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
        make_dataset_generator_for_lstmcnn_model(
            fn_draw_chart, window_size=window_size, predict_size=predict_size,
            path_to_csv=path_to_csv, dataset_type=dataset_type
        ),
        output_types=((tf.float64, tf.float32), tf.float16),
        output_shapes=((tf.TensorShape([29, 2]), tf.TensorShape((112, 112, 3))), tf.TensorShape([]))
    )
    return dataset

if __name__ == "__main__":
    if not ENABLE_IMAGE_DECODE:
        generate_chart_dataset(TRAIN_SP500_DATA_FILE, 'train')
    else:
        logging.error("Please set ENABLE_IMAGE_DECODE=False in the setting file")

