from sr import *
# from '.opt' import *

data_loader_train = Loader(size=5, 
                batch_size = 2,
                load_type = "nii.gz",
                subset='train', 
                images_dir="/opt/notebooks/denoise_ct/dataset",
                caches_dir="/opt/notebooks/denoise_ct/dataset/caches",
                percent=0.7)

# train = data_loader_train.dataset(1, random_transform=True)
train = data_loader_train.get_elements


data_loader_valid = Loader(size=5, 
                batch_size = 2,
                load_type = "nii.gz",
                subset='valid', 
                images_dir="/opt/notebooks/denoise_ct/dataset",
                caches_dir="/opt/notebooks/denoise_ct/dataset/caches",
                percent=0.7)

valid = data_loader_valid.get_elements
# valid = data_loader_valid.dataset(1, random_transform=True, repeat_count=1)


"""
trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
trainer.train(train,
                  valid.take(2),
                  steps=100, 
                  evaluate_every=10, 
                  save_best_only=False)

trainer.model.save_weights(weights_file('pre_generator.h5'))
"""

# %% [markdown]
# ### Treinando

# %%

unet = Gerador_UNet()
# unet_descriminador = descriminador()

generator = unet.generator()

# generator.load_weights(weights_file('pre_generator.h5'))
trainer = GeneratorTrainer(model=generator, loader = data_loader_train, checkpoint_dir=f'.ckpt/seg_pre_generator1{int(not training)}')
trainer.train(train,
                valid,
                # steps=1000000, 
                steps=1000,                                                                                                                              
                # evaluate_every=10000, 
                evaluate_every=10, 
                save_best_only=False)
# else: 
trainer.model.save_weights(weights_file('pre_generator.h5'))

def resolve_and_plot(noise_image_path):
    noise, original = noiser_np(noise_image_path)
    
    gan_sr = resolve_single(generator, noise)
    
    plt.figure(figsize=(20, 20))
    
    images = [noise, original, gan_sr]
    titles = ['noiser', "original",  'denoiser (GAN)']
    positions = [1, 2, 3]
    
    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

# %%
# resolve_and_plot('/opt/notebooks/dataset/Projeto/LR/4.png')

def __main__():
    pass