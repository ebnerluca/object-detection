from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils

with Image.open("./PennFudanPed/PedMasks/FudanPed00001_mask.png") as img:
    mask = np.array(img)
    img_scaled = np.floor(mask / np.max(mask) * 255).astype(np.uint8)
    plt.imshow(img_scaled)
    plt.show()

# Aleksandar's code below
mask = cv2.imread("./PennFudanPed/PedMasks/FudanPed00001_mask.png")
mask_scaled = np.floor(mask / np.max(mask) * 255).astype(np.uint8)

img = cv2.imread("./PennFudanPed/PNGImages/FudanPed00001.png")
utils.display_seg_mask(img[::4,::4,:], mask_scaled[::4,::4,:])
