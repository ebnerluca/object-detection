import os
import numpy as np
import torch
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    """
    TODO (David):
    * need to implement __len__ and __getitem__ methods
    * provide image and target using __getitem__
    * image is a PIL image of size (H, W)

    * target is a dict containing the fields below:

    - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from
    0 to W and 0 to H

    - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.

    - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset,
    and is used during evaluation

    ?? - area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric,
    to separate the metric scores between small, medium and large boxes.

    ?? - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.

    """
    def __init__(self, root, transforms):
        self.root = root  # path to dataset folder
        self.transforms = transforms  # ? probably not necessary

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))  # ToDo adjust to our dataset

    def __getitem__(self, idx):
        # access items like this: dataset[index]
        # returns image, target as specified above
        # TODO adjust tutorial code to our dataset & specifications
        pass

    def __len__(self):
        return len(self.imgs)
