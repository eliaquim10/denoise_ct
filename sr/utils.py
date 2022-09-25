from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
from .variaveis import *
import nibabel as nib

import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

Loader_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x, rgb_mean=Loader_RGB_MEAN):
    return (x - rgb_mean) / 127.5

def denormalize(x, rgb_mean=Loader_RGB_MEAN):
    return x * 127.5 + rgb_mean

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
    

# %% [markdown]
# #### Metricas

# %%
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

# %% [markdown]
# #### Transformacões

# %%
# @tf.function()
def load(entrada, saida):
    # Read and decode an image file to a uint8 tensor
    try:
        # print("aqui")
        # print(tf.decode_raw(entrada.numpy()))
        file_entrada =  bytes.decode(entrada.numpy())
        file_saida =  bytes.decode(saida.numpy())

        file_entrada = nib.load(file_entrada).get_fdata()
        file_saida = nib.load(file_saida).get_fdata()
        # print(file_entrada.shape, file_saida.shape)
        return tf.constant(file_entrada), tf.constant(file_saida)
    except:
        print("erro")
        return None
        
def load_lazy(entrada):
    try:
        # print("="*100)
        # print("__dict__", entrada.__dict__)
        file_entrada = nib.load(entrada).get_fdata()
        return file_entrada.transpose((2, 0, 1))
    except:
        print("erro")
        return None

def find_bound_box(filename):
    import xml.etree.ElementTree as ET
    #parsing XML file

    tree = ET.parse(filename)
    #fetching the root element

    root = tree.getroot()
    bound_box = []
    for object in root.findall('object'):
        name_element = object.find("name") 
        bndbox = object.find("bndbox") 
        xmin, ymin, xmax, ymax = bndbox.find("xmin").text, bndbox.find("ymin").text, bndbox.find("xmax").text, bndbox.find("ymax").text
        name = name_element.text    

        # print(xmin, ymin, xmax, ymax)
        bound_box.append({"name": name,  "bbox":[int(xmin), int(ymin), int(xmax), int(ymax)]})

        # print(subelem.tag, object.attrib, subelem.text)

    return bound_box


def load_image(path):
    image = np.array(Image.open(path))
    return tf.cast(image, tf.float32)

# @tf.function()
def inputr(img, std):

    new_img = tf.cast(img, tf.float32)
    new_img = new_img / 255

    input = tf.random.normal(shape = new_img.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32) 
    input_img = new_img + input
    input_img = tf.clip_by_value(input_img, 0.0, 1.0)
    return input_img, new_img

def inputr_np(filename):
    img = np.array(Image.open(filename))
    input = np.random.normal(127, 255, img.get_shape())
    input_img = img + input
    input_img = tf.clip_by_value(input_img, 0., 255.)

    return input_img, img

# @tf.function()
def random_crop(x, y):

    # print("====", x.get_shape(), y.get_shape())
    y_shape = x.get_shape()[:2]
    
    input_crop_size = [256, 256]
    # input_crop_size[1] =  y_shape[1] // scale 
    # input_crop_size[0] =  y_shape[0] // scale
    # print(input_crop_size)

    y_w = tf.random.uniform(shape=(), maxval=y_shape[1] - input_crop_size[1] + 1, dtype=tf.int32)
    y_h = tf.random.uniform(shape=(), maxval=y_shape[0] - input_crop_size[0] + 1, dtype=tf.int32)

    x_cropped = x[y_h:y_h + input_crop_size[0], y_w:y_w + input_crop_size[1]]
    y_cropped = y[y_h:y_h + input_crop_size[0], y_w:y_w + input_crop_size[1]]

    return x_cropped, y_cropped 

@tf.function()
def random_flip(input_img, target_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (input_img, target_img),
                   lambda: (tf.image.flip_left_right(input_img),
                            tf.image.flip_left_right(target_img)))

@tf.function()
def random_rotate(input_img, target_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(input_img, rn), tf.image.rot90(target_img, rn)

@tf.function()
def one_hot(target):
    """
        0: Background (None of the following organs)
        1: Liver
        2: Bladder
        3: Lungs
        4: Kidneys
        5: Bone
        6: Brain
    """
    indices = [1, 2, 3, 4, 5, 6]
    target = tf.cast(target, dtype=tf.uint8)
    target = tf.one_hot(target, len(indices))
    return target
# %%
def resolve_single(model, input):
    return resolve(model, tf.expand_dims(input, axis=0))[0]


def resolve(model, input_batch):
    input_batch = tf.cast(input_batch, tf.float32)
    target_batch = model(input_batch)
    target_batch = tf.clip_by_value(target_batch, 0, 255)
    target_batch = tf.round(target_batch)
    target_batch = tf.cast(target_batch, tf.uint8)
    return target_batch

def cast(target_batch):
    target_batch = tf.clip_by_value(target_batch, 0, 255)
    target_batch = tf.round(target_batch)
    target_batch = tf.cast(target_batch, tf.uint8)
    return target_batch

def resolve_mae(model, input_batch):
    input_batch = tf.cast(input_batch, tf.float32)
    target_batch = model(input_batch)
    target_batch = tf.clip_by_value(target_batch, 0, 255)
    target_batch = tf.round(target_batch)
    return target_batch


def evaluate(model, dataset):
    psnr_values = []
    for input, real in dataset:
        target = resolve(model, input)

        psnr_value = psnr(real, target)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

def evaluate_mae(model, dataset):
    mae_values = []
    for input, real in dataset:
        target = resolve_mae(model, input)

        mae = metric_mae(real, target)
        mae_values.append(mae)
    return tf.reduce_mean(mae_values)

weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

def resolve_single(model, target):
    return resolve(model, tf.expand_dims(target, axis=0))[0]
    
# def load_image(path):
#     return np.array(Image.open(path))


def plot_sample(target, sr):
    plt.figure(figsize=(20, 10))

    images = [target, sr]
    titles = ['target', f'SR (x{sr.shape[0] // target.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)

def __main__():
    pass
if __name__ == "__main__":
    pass