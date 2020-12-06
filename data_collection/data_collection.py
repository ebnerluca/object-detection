import numpy as np
import cv2
from agent import PurePursuitPolicy
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import matplotlib.pyplot as plt # for visualisation

"""
    Open questions:
    - do we need to include the background class as it's not there in the
    examples?
    - how should we make the dataset images 244x244x3, crop or squash?
    - skip a few frames for dataset
"""

debug = False
npz_index = 0
def save_npz(img, boxes, classes):
    """ Save the non-segmented observation, boxes and classes of one instant
        into an npz file in dataset directory.
        img = 244x244x3 array
        boxes = list of [xmin, ymin, xmax, ymax]
        labels = np array corresponding to boxes
    """

    global npz_index
    with makedirs("./data_collection/dataset"):
        np.savez(f"./data_collection/dataset/{npz_index}.npz", *(img, boxes, classes))
        print(f"Saved {npz_index}.npz")
        npz_index += 1


def clean_segmented_image(seg_img):
    """ Steps:
        - split the image into 5 images consisting of each class
        - identify classes
        - clean the noise (use findContours?)
        - make bounding boxes for each segmentation
        note that seg_img is probably intended to be in the RGB color space
        as the simulation renders it the same as plt.imshow()
    """

    # split image by color in HSV, from 
    # https://pinetools.com/image-color-picker, note these values need to scaled
    # to fit cv2 HSV. online range: (359,99,99), cv2 range: (179,255,255)
    # 0: background is (300,100,100) (pink) 
    # 1: duckie is (231.9,56,66) to (231.5,56,89) (blue)
    # 2: cone is (4.8,55,44) to (4.8,56,89) (coral)
    # 3: truck is (280,3,46) to (258,3,) (gray)
    # 4: bus is (46.6,93,85) (yellow)
    # try without tolerance (2nd value) first

    color_ranges = {
        0: {'low': (149, 254, 254), 'high': (151, 256, 256)},
        1: {'low': (110, 140, 170), 'high': (120, 150, 255)},
        2: {'low': (2, 141, 113), 'high': (3, 145, 230)},
        3: {'low': (139, 6, 116), 'high': (141, 26, 120)},
        4: {'low': (22, 236, 215), 'high': (24, 238, 217)}
    }

    if debug: # show seg_img
        plt.imshow(seg_img)
        plt.title("Segmented image")
        plt.show()

    seg_img_hsv = cv2.cvtColor(seg_img, cv2.COLOR_RGB2HSV)

    # to find ranges use:
    # plt.imshow(seg_img_hsv)
    # plt.show() # then point mouse at object

    boxes = []
    classes = []
    for class_ in range(1, 5): # skip background class
        # find mask for class color range
        mask = cv2.inRange(seg_img_hsv, color_ranges[class_]['low'],color_ranges[class_]['high'])

        if debug: # shows the masks
            result = cv2.bitwise_and(seg_img.copy(), seg_img, mask=mask)
            plt.imshow(result)
            plt.title(f"Mask of class {class_}")
            plt.show()

        # find the contours (ignoring max noise area)
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 2] # note min noise area is 10

        if debug: # shows the contours
            img = cv2.drawContours(seg_img.copy(), contours, -1, (0,255,0), 3)
            plt.imshow(img)
            plt.title(f"Contours for class {class_}")
            plt.show()

        # find the bounding boxes for each contour
        for contour in contours:
            if debug: # shows the contour
                img = cv2.drawContours(seg_img.copy(), contour, -1, (0,255,0), 3)
                plt.imshow(img)
                plt.title(f"Contour for class {class_}")
                plt.show()

            xmin, ymin, width, height = cv2.boundingRect(contour)
            xmax, ymax = (xmin + width, ymin + height)
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            classes.append(class_)

            if debug: # show the boxes
                img = cv2.rectangle(seg_img.copy(), tuple(box[0:2]),tuple(box[2:4]) , (0,255,0) , 2)
                plt.imshow(img)
                plt.title(f"Box for class {class_}")
                plt.show()
            

    classes = np.array(classes)

    return boxes, classes


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

SAMPLE_FREQ = 10

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # resize images to 244x244x3, ready for dataset
        # height, width, channels = np.shape(obs)
        # left_border = int(width/2 - height/2)
        # right_border = int(width/2 + height/2)
        # obs = obs[:, left_border:right_border, :]  # cut off sides to make img square
        # segmented_obs = segmented_obs[:, left_border:right_border, :]  # cut off sides to make img square

        obs = cv2.resize(obs, (224, 224))
        segmented_obs = cv2.resize(segmented_obs, (224, 224))

        if nb_of_steps % SAMPLE_FREQ == 0:
            boxes, classes = clean_segmented_image(segmented_obs)
            save_npz(obs, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
