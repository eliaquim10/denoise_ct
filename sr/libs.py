from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, ReLU, Concatenate, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu

# from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

import time
import tensorflow as tf
import datetime


# from model import evaluate
# from model import srgan

from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.losses import MeanAbsoluteError
# from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


from tensorflow.python.data.experimental import AUTOTUNE
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import trange, tqdm
import nibabel as nib

# %matplotlib inline

import argparse

def __main__():
    pass
if __name__ == "__main__":
    pass