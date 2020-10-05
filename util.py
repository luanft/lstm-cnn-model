"""Utilily functions for models"""
import tensorflow as tf
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