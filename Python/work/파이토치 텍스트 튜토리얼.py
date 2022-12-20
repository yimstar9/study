import torchtext
print(torchtext.__version__)
torchtext.data

import tensorflow as tf
tf.__version__
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import torch
print("PyTorch version: {}".format(torch.__version__))
print("CUDA version: {}".format(torch.version.cuda))
print(torch.cuda.get_device_name(0))

import torchtext
print(torchtext.__version__)
from torchtext.datasets import IMDB
import urllib.request
import pandas as pd
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
df.head()
IMDB('.data','train','test')