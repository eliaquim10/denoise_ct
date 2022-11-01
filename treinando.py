from sr import *
# from '.opt' import *


# def treinando(args):
parser = argparse.ArgumentParser()

parser.add_argument("--size", nargs="?", type=int, default=5)
parser.add_argument("--batch_size", nargs="?", type=int, default=2)
parser.add_argument("--load_type", nargs="?", type=str, default="npy") #nii.gz
parser.add_argument("--images_dir", nargs="?", type=str, default="/opt/notebooks/denoise_ct/dataset/v1")
parser.add_argument("--caches_dir", nargs="?", type=str, default="/opt/notebooks/denoise_ct/dataset/caches")
parser.add_argument("--weights", nargs="?", type=str, default="")

args = parser.parse_args()

data_loader_train = Loader(size=args.size, 
                batch_size = args.batch_size,
                load_type = args.load_type,
                subset='train', 
                images_dir=args.images_dir,
                caches_dir=args.caches_dir,
                percent=0.7)

# train = data_loader_train.dataset(1, random_transform=True)
train = data_loader_train.get_elements


data_loader_valid = Loader(size=args.size, 
                batch_size = args.batch_size,
                load_type = args.load_type,
                subset='valid', 
                images_dir=args.images_dir,
                caches_dir=args.caches_dir,
                percent=0.7)

valid = data_loader_valid.get_elements
# valid = data_loader_valid.dataset(1, random_transform=True, repeat_count=1)


unet = Gerador_UNet()

# generator = unet.generator()
generator = unet.generator_modify() # _modify


# generator.load_weights(weights_file('pre_generator.h5'))
trainer = GeneratorTrainer(model=generator, 
                # loss = SparseCategoricalCrossentropy(),                                                                                                                              
                # loss = CategoricalCrossentropy(),                                                                                                                              
                loss = BinaryCrossentropy(),                                                                                                                              
                checkpoint_dir='.ckpt/seg_pre_generator1', #from_logits = True
                # learning_rate=1e-3
                learning_rate=PiecewiseConstantDecay(boundaries=[1, 45, 50], values=[1e-2, 1e-5, 1e-7, 1e-7])  #, 1e-6
                )
trainer.train(train,
                valid,
                # steps=1000000, 
                steps=3,
                # evaluate_every=10000, 
                evaluate_every=1, 
                save_best_only=False)

print("salvando o modelo")
trainer.model.save_weights(weights_file(args.weights, 'pre_generator.h5'))

# def resolve_and_plot(noise_image_path):
#     noise, original = noiser_np(noise_image_path)
    
#     gan_sr = resolve_single(generator, noise)
    
#     plt.figure(figsize=(20, 20))
    
#     images = [noise, original, gan_sr]
#     titles = ['noiser', "original",  'denoiser (GAN)']
#     positions = [1, 2, 3]
    
#     for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
#         plt.subplot(2, 2, pos)
#         plt.imshow(img)
#         plt.title(title)
#         plt.xticks([])
#         plt.yticks([])


# resolve_and_plot('/opt/notebooks/dataset/Projeto/LR/4.png')

def __main__():
    pass

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--size", nargs="?", type=int, default=5)
#     parser.add_argument("--batch_size", nargs="?", type=int, default=2)
#     parser.add_argument("--load_type", nargs="?", type=str, default="npy") #nii.gz
#     parser.add_argument("--images_dir", nargs="?", type=str, default="/opt/notebooks/denoise_ct/dataset")
#     parser.add_argument("--caches_dir", nargs="?", type=str, default="/opt/notebooks/denoise_ct/dataset/caches")
#     parser.add_argument("--weights", nargs="?", type=str, default="")

#     args = parser.parse_args()

#     treinando(args)
# python treinando.py --size 5 --batch_size 2 --load_type nii.gz2 --images_dir /opt/notebooks/denoise_ct/dataset --caches_dir /opt/notebooks/denoise_ct/dataset/caches --weights 
