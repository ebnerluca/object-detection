import numpy as np
import cv2

# dataset_folder = "./data_collection/dataset/"  # assuming you run from within repo root folder
dataset_folder = "./newstyle_validation_dataset/"
print(f"Inspecting data from folder {dataset_folder}...")
min_idx = 0
max_idx = 1999

def get_color(class_idx):

    color = {
        0: (0, 0, 0),  # background
        1: (0, 255, 0),  # duckie
        2: (100, 100, 255),  # cone
        3: (255, 150, 50),  # truck
        4: (255, 50, 150)  # bus
    }
    return color.get(class_idx, (255, 255, 255))

def inspect_image_and_boxes(npzfile):

    img = npzfile['arr_0']
    boxes = npzfile['arr_1']
    classes = npzfile['arr_2']

    # convert color space to visualize with cv2.imshow
    img_rgb = img.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # show raw image
    cv2.imshow("raw_image", cv2.resize(img_bgr, (1000, 1000)))

    # show img with boxes
    img_boxed = img_bgr.copy()
    for i in range(len(boxes)):
        img_boxed = cv2.rectangle(img_boxed, tuple(boxes[i][0:2]), tuple(boxes[i][2:4]), get_color(classes[i]), 2)
    cv2.imshow("boxed_image", cv2.resize(img_boxed, (1000,1000)))
    print(f"Found {len(boxes)} boxes")
    cv2.waitKey(0)


for i in range(min_idx, max_idx):
    file_name = f"{i}.npz"
    path = dataset_folder + file_name
    npzfile = np.load(path)
    print(f"Inspecting {file_name}...")
    inspect_image_and_boxes(npzfile)



