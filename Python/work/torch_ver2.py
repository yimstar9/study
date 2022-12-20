import torch
import torch
torch.cuda.get_device_name(0)
torch.cuda.is_available()
print(torch.__version__)

import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.__version__