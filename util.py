"""Utilily functions for models"""
import os
import datetime
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import mean_absolute_error
import setting


def batch_norm(input):
    """batch normalize for the vector"""
    return tf.keras.layers.BatchNormalization()(input)

def convert_to_list(variant_dataset):
    """Convert data set to list of value"""
    flatten_dataset: tf.data.Dataset = variant_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    return list(flatten_dataset.as_numpy_iterator())

if setting.ENABLE_IMAGE_DECODE:
    def decode_img(file_path):
        """decode image"""
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        return tf.image.resize(img, [112, 112])
else:
    def decode_img(file_path):
        """decode image"""
        return 0

def get_cache_file(ctype, mtype):
    """return the cache file for cache type and model"""
    container_dir = os.path.join(setting.CACHE_DIR, ctype, mtype)
    os.makedirs(container_dir, exist_ok=True)
    return os.path.join(container_dir, "%s_%s.cache" % (ctype, mtype))


def get_log_dir(mtype, tname=None):
    """return the log dir for cache type and model"""
    log_dir = tname
    if tname is None:
        log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    container_dir = os.path.join(setting.CACHE_DIR, "logs", mtype, log_dir)
    tf.print("Logging to dir")
    tf.print(log_dir)
    os.makedirs(container_dir, exist_ok=True)
    return container_dir


def set_gpu_memory_limit(gpu_index, memory_limit):
    """Set the memory limit"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_index],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)



def RMAE(y_true, y_pred):
    """RMAE"""
    return K.sqrt(mean_absolute_error(y_true, y_pred))

