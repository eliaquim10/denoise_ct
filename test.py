# from . import *
# from '.opt' import *

# data_loader_train = Loader(size=4, 
#                 batch_size = 4,
#                 subset='train', 
#                 images_dir="OrganSegmentations",
#                 caches_dir="dataset/caches",
#                 percent=0.7)

# # train = data_loader_train.dataset(1, random_transform=True)
# train = data_loader_train.load()

# for file_entrada, file_target in enumerate(train):
#     dataset = data_loader_train.loader.load_file(entrada = file_entrada, target = file_target)
import numpy as np
a = np.ones((3,3,3))
print(a[1:].shape)