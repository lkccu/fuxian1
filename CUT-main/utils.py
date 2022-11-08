import os
import tensorflow as tf


def create_dir(dir):
    """ Create the directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir

@tf.function
def load_image(image_file, image_size=None, data_augmentation=True):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if data_augmentation:
        image = tf.image.random_flip_left_right(image)
    if image_size is not None:
        image = tf.image.resize(image, size=(image_size[0], image_size[1]))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1,1,3])

    return  image
