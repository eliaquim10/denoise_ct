from sr import *
# from '.opt' import *

data_loader_train = Loader(size=10, 
                subset='train', 
                images_dir="/opt/notebooks/tcc/OrganSegmentations",
                caches_dir="/opt/notebooks/tcc/dataset/caches",
                percent=0.7)

train = data_loader_train.dataset(4, random_transform=True)


data_loader_valid = Loader(size=10, 
                subset='train', 
                images_dir="/opt/notebooks/tcc/OrganSegmentations",
                caches_dir="/opt/notebooks/tcc/dataset/caches",
                percent=0.7)
valid = data_loader_valid.dataset(4, random_transform=True, repeat_count=1)


"""
pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir=f'.ckpt/pre_generator')
pre_trainer.train(train,
                  valid.take(2),
                  steps=100, 
                  evaluate_every=10, 
                  save_best_only=False)

pre_trainer.model.save_weights(weights_file('pre_generator.h5'))
"""

# %% [markdown]
# ### Treinando

# %%

training = True
gan_training = True
unet = Gerador_UNet()
unet_descriminador = descriminador()

gan_generator = unet.generator()
# gan_generator = generator(num_filters = 32, num_res_blocks=8)
# discriminador = unet_descriminador.Discriminator()
# discriminador = discriminator(16)

# gan_generator.load_weights(weights_file('pre_generator.h5'))
# print(gan_generator.summary())

# valor = tf.cast(np.ones((1, 256, 256, 3)), tf.float32)
# output = gan_generator(valor, training=False)
# print(tf.shape(output))
# print(output)

if (training):
    pre_trainer = GeneratorTrainer(model=gan_generator, checkpoint_dir=f'.ckpt/seg_pre_generator1{int(not training)}')
    pre_trainer.train(train,
                    valid.take(4),
                    # steps=1000000, 
                    steps=1000,                                                                                                                              
                    # evaluate_every=10000, 
                    evaluate_every=10, 
                    save_best_only=False)
# else: 
    # gan_generator.load_weights(weights_file('pre_generator.h5'))
    pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

if(not training and gan_training):
    gan_trainer = GanTrainer(generator=gan_generator, 
                                discriminator=discriminador, 
                                Lambda=opt.Lambda, 
                                g_learning_rate=opt.glr,
                                g_loss = unet.loss_generador, 
                                d_learning_rate=opt.dlr,
                                d_loss = unet_descriminador.loss_discriminator)
    gan_trainer.train(train, steps=200000)
    gan_generator.save_weights(weights_file('gan_generator.h5'))

# %%
# pre_generator = generator()
# gan_generator = generator()

# pre_generator.load_weights(weights_file('pre_generator.h5'))
# gan_generator.load_weights(weights_file('gan_generator.h5'))

# %%
# from model import resolve_single
# from utils import load_image

def resolve_and_plot(noise_image_path):
    noise, original = noiser_np(noise_image_path)
    
    gan_sr = resolve_single(gan_generator, noise)
    
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