import tensorflow as tf
tf.__version__
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

print(tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()