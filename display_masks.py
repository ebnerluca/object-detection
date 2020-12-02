from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

with Image.open("../PennFudanPed/PedMasks/FudanPed00001_mask.png") as img:
    mask = np.array(img)
    img_scaled = np.floor(mask / np.max(mask) * 255).astype(np.uint8)
    plt.imshow(img_scaled)
    plt.show()
