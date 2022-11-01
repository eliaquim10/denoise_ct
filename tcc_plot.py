
from sr.gerador import Gerador_UNet
from sr.grad_cam import GradCAM
from sr.utils import input_nib_l, input_nib_v, one_hot, slice_lung, resolve_single, weights_file, random_crop
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

unet = Gerador_UNet()
log_dir = 'logs/graph/'
graph = tf.summary.create_file_writer(log_dir)

generator = unet.generator_modify() #_modify
# generator.load_weights(weights_file(None, 'pre_generator.h5'))
# generator.load_weights("C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\weights\\srgan\\pre_generator.h5")
generator.load_weights("C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\weights\\srgan\\pre_generator.h5")

# print(generator.last)
generator.summary()

def resolve_and_plot(paciente, slice, slice_file):

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

    # dir_input = f"/opt/notebooks/denoise_ct/dataset/v1/volume/patient_{paciente}_slice_{slice_file}.nii.gz"
    # dir_label = f"/opt/notebooks/denoise_ct/dataset/v1/label/patient_{paciente}_slice_{slice_file}.nii.gz"

    dir_input = f"dataset//v1//volume//patient_{paciente}_slice_{slice_file}.nii.gz"
    dir_label = f"dataset//v1//label//patient_{paciente}_slice_{slice_file}.nii.gz"

    entrada, seg = input_nib_v(dir_input, slice), input_nib_l(dir_label, slice)
    entrada, seg = random_crop(entrada, seg, 256)
    seg = slice_lung(seg)
    # entrada = tf.clip_by_value(entrada, 0., 1.)

    # tf.summary.trace_on(graph=True) # , profiler=True
    
    # graph
    # exit()
    # with graph.as_default():    
    #     with tf.device("/device:cpu:0"):    
    gen_seg = resolve_single(generator, entrada)

    i = np.argmax(gen_seg[0])
    icam = GradCAM(generator, i, "concatenate_3")
    # icam = GradCAM(generator, i, "last_layer")

    heatmap = icam.compute_heatmap(entrada)
    # heatmap = icam.make_gradcam_heatmap(entrada)
    
    # heatmap = cv2.resize(heatmap, (32, 32))

    
    print(heatmap.shape, entrada.shape)
    # heatmap = icam.resize(heatmap, entrada)

    # (heatmap, output) = icam.overlay_heatmap(heatmap, entrada, alpha=0.5)
    (heatmap, output) = icam.display_gradcam(heatmap, entrada, alpha=0.5)

    # fig, ax = plt.subplots(1, 3)

    # ax[0].imshow(heatmap)
    # ax[1].imshow(entrada)
    # ax[2].imshow(output)
    # tf.summary.trace_export(
    #     name="resolve_single_trace",
    #     step=0,
    #     profiler_outdir=log_dir)
    # gen_seg = gen_seg[:,:, 3]
    # print(entrada)
    print(heatmap.shape)
    print(seg.shape)
    print(gen_seg)
    
    plt.figure(figsize=(16, 16))
    lista = list(map(lambda num: (seg[:,:, num], f"seg_{num+1}", 2*(num + 1) + 3) , range(seg.shape[-1])))
    lista = lista + list(map(lambda num: (gen_seg[:,:, num], f"gen_seg_{num+1}", 2*(num + 1) + 4) , range(seg.shape[-1])))
    lista = [(entrada, "original", 1)] + lista
    lista = [(heatmap, "heatmap", 2)] + lista
    lista = [(output, "output", 3)] + lista
    tamanho = len(lista)
    print("tamanho=", tamanho)

    # figure, axes = plt.subplot(nrows=tamanho //2 + 1, ncols = 2)
    # soma = tf.reduce_sum(seg[:,])
    # if soma < 0:
    #     return
    for i, (img, title, pos) in enumerate(lista):
        # print(title, img[0,0])
        plt.subplot(tamanho // 2 + 2, 2, pos)
        plt.imshow(img) # , cmap='gray'
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

    plt.show()
# %%
# resolve_and_plot(6, -5)

resolve_and_plot(100, 8, 650)

# resolve_and_plot(0, 12, 50)
# patient_0_slice_50.nii.gz 12
# C:\Users\emn3\Documents\workspace\seg\denoise_ct\dataset\v1\volume\patient_100_slice_650.nii.gz
# resolve_and_plot(0, 0, 0)
# /opt/notebooks/denoise_ct/dataset/v1/label/patient_132_slice_250.nii.gz
