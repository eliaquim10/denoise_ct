dir_label = "/opt/notebooks/denoise_ct/OrganSegmentations/labels-0.nii.gz"
dir_volume = "/opt/notebooks/denoise_ct/OrganSegmentations/volume-0.nii.gz"

from matplotlib import pyplot as plt

from utils import load_lazy

img_label = load_lazy(dir_label)
img_volume = load_lazy(dir_volume)

import numpy as np

print("Tipo img_label:", type(img_label))
print("Shape label image:", img_label.shape)
print("Shape volume image:", img_volume.shape)
print("min(img_label):", np.min(img_label), "| max(img_label):", np.max(img_label))
print("min(img_volume):", np.min(img_volume), "| max(img_volume):", np.max(img_volume), "\n\n")

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot('121').set_title('label image')
imgplot = plt.imshow(img_label, cmap='gray')
ax = fig.add_subplot('122').set_title('volume image')
imgplot = plt.imshow(img_volume, cmap='gray')
fig.show()

def __main__():
    pass
if __name__ == "__main__":
    pass