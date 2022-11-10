
from sr.gerador import Gerador_UNet, FCN32, UNET
from sr.grad_cam import GradCAM
from sr.perda import DiceLoss, FocalLoss
from sr.utils import input_nib_l, input_nib_v, one_hot, slice_lung, resolve_single, weights_file, random_crop, plot_sample, one_hot_lung
# from matplotlib import pyplot as plt
import tensorflow as tf
# import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.metrics import MeanIoU

def resolve_and_plot(paciente, slice):
    unet_use = True
    cam = False
    if unet_use:
        unet = Gerador_UNet()
        log_dir = 'logs/graph/'
        graph = tf.summary.create_file_writer(log_dir)

        generator = unet.generator_modify() #_modify
        # generator = UNET(2)
        # generator.load_weights(weights_file(None, 'pre_generator.h5'))
        # generator.load_weights("C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\weights\\srgan\\pre_generator.h5")
        # generator.load_weights("weights\\gen_50\\pre_generator.h5")
        generator.load_weights("weights\\gen_bce_60\\unet_pre_generator.h5")
        # generator.load_weights("weights\\srgan\\unet_pre_generator.h5")

        # print(generator.last)
    else:
        generator = FCN32(2)
    # print("generator", generator.layers[0].name)
    print(generator(tf.zeros((1,128,128,1))).shape)
    generator.summary()

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

    # dir_input = f"dataset//v1//volume//patient_{paciente}_slice_{slice_file}.nii.gz"
    # dir_label = f"dataset//v1//label//patient_{paciente}_slice_{slice_file}.nii.gz"

    
    dir_input = f"OrganSegmentations//volume-{paciente}.nii.gz"
    dir_label = f"OrganSegmentations//labels-{paciente}.nii.gz"

    entrada, seg = input_nib_v(dir_input, slice), input_nib_l(dir_label, slice)
    entrada, seg = random_crop(entrada, seg, 128)
    
    entrada = tf.expand_dims(entrada, 2)
    if not unet_use:
        entrada = tf.concat([entrada, entrada, entrada], 2)

    print(entrada.shape)
    
    seg = slice_lung(seg)

    seg = one_hot_lung(seg)
    
    # entrada = tf.clip_by_value(entrada, 0., 1.)

    # tf.summary.trace_on(graph=True) # , profiler=True
    
    # graph
    # exit()
    # with graph.as_default():    
    #     with tf.device("/device:cpu:0"):    
    gen_seg = resolve_single(generator, entrada)
    # show_heatmap = False
    i = np.argmax(gen_seg[0])
    icam = GradCAM(generator, i)  # , "model"# , "concatenate_3" "last_layer"
    if cam:
        # heatmap = icam.compute_heatmap(entrada)
        heatmap = icam.make_gradcam_heatmap(entrada)
        print("heatmap", heatmap.shape)
        
        # (heatmap, output) = icam.compute_heatmap_unet(entrada, alpha=0.5)
        # (heatmap, output) = icam.compute_heatmap(entrada, alpha=0.5)
        (heatmap, output) = icam.overlay_heatmap(heatmap, entrada, alpha=0.5)

    
    # print(heatmap.shape)
    print(seg.shape)
    # print(gen_seg.shape)
    
    def show_img(gen_seg):
        img = tf.math.argmax(gen_seg,axis=2)
        img = icam.array_img(img.numpy())
#         img = give_color_to_annotation(img)
        return img
    dice_loss = DiceLoss(class_indexes=2)
    focal_loss = FocalLoss()
    meanIoU = MeanIoU(num_classes = 2)
    # print("dice", dice_loss.dice_coef(seg, gen_seg))
    print("meanIoU", meanIoU(seg, gen_seg).numpy())
    print("dice_loss", dice_loss(seg, gen_seg).numpy())
    print("dice", dice_loss.dice(seg, gen_seg).numpy())
    print("focal_loss", focal_loss(seg, gen_seg).numpy())
    
    # dice_coef
    # lista = list(map(lambda num: (seg[:,:, num], f"seg_{num+1}", 2*(num + 1) + 3) , range(seg.shape[-1])))
    # lista = lista + list(map(lambda num: (gen_seg[:,:, num], f"gen_seg_{num+1}", 2*(num + 1) + 4) , range(seg.shape[-1])))
    lista = [(show_img(seg),"seg",  5)]    
    lista = lista + [(show_img(gen_seg),"gen_seg",  6)]
    
    lista = [(entrada, "original", 1)] + lista
    if cam:
        lista = [(heatmap, "heatmap", 2)] + lista
        lista = [(output, "output", 3)] + lista
    # tamanho = len(lista)
    # print("tamanho=", tamanho)

    plot_sample(lista)
    
# %%
# resolve_and_plot(6, -5)

# resolve_and_plot(100, 8, 650)

# resolve_and_plot(0, 12, 50)
with tf.device("/GPU:0"):
    resolve_and_plot(0, 73)
# patient_0_slice_50.nii.gz 12
# C:\Users\emn3\Documents\workspace\seg\denoise_ct\dataset\v1\volume\patient_100_slice_650.nii.gz
# resolve_and_plot(0, 0, 0)
# /opt/notebooks/denoise_ct/dataset/v1/label/patient_132_slice_250.nii.gz
