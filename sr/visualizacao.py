dir_lr = "/opt/notebooks/dataset/Projeto/LR/1.png"
dir_hr = "/opt/notebooks/dataset/Projeto/HR/1.png"

from matplotlib import pyplot as plt
from matplotlib import image

img_lr = image.imread(dir_lr)
img_hr = image.imread(dir_hr)

import numpy as np

print("Tipo img_lr:", type(img_lr))
print("Shape LR image:", img_lr.shape)
print("Shape HR image:", img_hr.shape)
print("min(img_lr):", np.min(img_lr), "| max(img_lr):", np.max(img_lr))
print("min(img_hr):", np.min(img_hr), "| max(img_hr):", np.max(img_hr), "\n\n")

fig = plt.figure(figsize=(16,16))
ax = fig.add_subplot('121').set_title('LR image')
imgplot = plt.imshow(img_lr, cmap='gray')
ax = fig.add_subplot('122').set_title('HR image')
imgplot = plt.imshow(img_hr, cmap='gray')
fig.show()

def __main__():
    pass
if __name__ == "__main__":
    pass