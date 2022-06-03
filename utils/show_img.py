from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def show_img_mask(img, mask, start_slice = 75):
    transparent1 = 0.8
    transparent2 = 1.0
    cmap = plt.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(10, 6))
    plt.suptitle("Mask on image", fontsize=18)
    start_slice = start_slice
    for i in range(0, 6):
        imgslice = img[i + start_slice]
        maskslice = mask[i + start_slice]
        ax = plt.subplot(2, 3, i + 1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ax.set_title('Slice {}'.format(i + start_slice))
        ax.axis('off')
        plt.imshow(imgslice, cmap='gray', alpha=transparent1)
        plt.imshow(maskslice, cmap=my_cmap, alpha=transparent2)
        plt.pause(0.001)
    plt.show()
    plt.close()

def show_img(img, start_slice = 75):
    plt.figure(figsize=(10, 6))
    plt.suptitle("Image", fontsize=18)
    start_slice = start_slice
    for i in range(0, 6):
        imgslice = img[i + start_slice]
        ax = plt.subplot(2, 3, i + 1)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ax.set_title('Slice {}'.format(i + start_slice))
        ax.axis('off')
        plt.imshow(imgslice, cmap='gray')
        plt.pause(0.001)
    plt.show()
    plt.close()