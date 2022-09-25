from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19

# from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

import time
import tensorflow as tf
import datetime


# from model import evaluate
# from model import srgan

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


from tensorflow.python.data.experimental import AUTOTUNE

import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
# %matplotlib inline

import argparse

def __main__():
    pass
if __name__ == "__main__":
    pass