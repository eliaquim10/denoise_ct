
import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib


from matplotlib import pyplot as plt

import numpy as np


# %% [markdown]
# ## UNET SEG

# %%
# from sr.gerador import Gerador_UNet
from sr.utils import input_nib_l, input_nib_v, slice_lung # one_hot, resolve_single, weights_file, random_crop
from matplotlib import pyplot as plt
import tensorflow as tf

# unet = Gerador_UNet()

# generator = unet.generator() #_modify
# # generator.load_weights(weights_file(None, 'pre_generator.h5'))
# generator.load_weights("C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\weights\\srgan\\pre_generator.h5")

# generator.summary()
def show(entrada, seg, gen_seg):
    plt.figure(figsize=(16, 16))
    lista = list(map(lambda num: (seg[:,:, num], f"seg_{num+1}", 2*(num + 1) + 1) , range(seg.shape[-1])))
    # lista = lista + list(map(lambda num: (gen_seg[:,:, num], f"gen_seg_{num+1}", 2*(num + 1) + 2) , range(seg.shape[-1])))
    lista = [(entrada, "original", 1)] + lista
    tamanho = len(lista)

    # figure, axes = plt.subplot(nrows=tamanho //2 + 1, ncols = 2)
    for _, (img, title, pos) in enumerate(lista):
        # print(title, img[0,0])
        plt.subplot(tamanho // 2 + 2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.show()
def resolve_and_plot(paciente, slice):

    """
        0: Background (None of the following organs)
        1: Liver (figado)
        2: Bladder (bexiga)
        3: Lungs (pulmao)
        4: Kidneys (rins)
        5: Bone (osso)
        6: Brain (cerebro)
        # amarelo = 1 e roxo = 0
    """

    dir_input = f"C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\v1\\volume\\{paciente}"
    dir_label = f"C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\v1\\label\\{paciente}"


   
    entrada, seg  = input_nib_v(dir_input, slice), input_nib_l(dir_label, slice)
    # entrada, seg = random_crop(entrada, seg, 256)

    seg = slice_lung(seg)
    
    with tf.device("/GPU:0"):
        soma = tf.reduce_sum(seg)
    if soma < 3000:
        return False
    print(soma.numpy())
    print(dir_input, slice)

    # gen_seg = resolve_single(generator, entrada)
    # gen_seg = gen_seg[:,:, 3]
    
    show(entrada, seg, None)
    return True
# %%
# resolve_and_plot(6, -5)
for file_name in os.listdir("C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\v1\\volume\\"):
    segundo_for = False
    size_slice = nib.load(f"C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\v1\\volume\\{file_name}").shape[-1]
    for i in range(size_slice):
        if resolve_and_plot(file_name, i):
            segundo_for = True
            break
    if segundo_for:
        break 



