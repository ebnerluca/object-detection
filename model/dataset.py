import os
import numpy as np
import torch
from PIL import Image, ImageShow


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
    # def __init__(self, root, transforms):
    def __init__(self, root):
        self.root = root  # path to dataset folder
        # self.transforms = transforms  # ? probably not necessary

        # Count dataset size
        self.dataset_size = 0
        fileidx = 0
        filename = f'{fileidx}.npz'
        while os.path.isfile(os.path.join(self.root, filename)):
            self.dataset_size += 1
            fileidx += 1
            filename = f'{fileidx}.npz'

        print(f'[Dataset]: Found dataset with {self.dataset_size} files.')

    def __getitem__(self, idx):
        # access items like this: dataset[index]
        # returns image, target as specified above

        filename = f'{idx}.npz'
        path = os.path.join(self.root, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(f'[Dataset]: File at {path} does not exist!')

        npzfile = np.load(path)

        img = Image.fromarray(npzfile['arr_0'])
        img = img.convert("RGB")
        boxes = npzfile['arr_1']
        classes = npzfile['arr_2']

        # convert lists to tensors:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(classes, dtype=torch.int64)
        image_id = torch.as_tensor(idx, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        return img, target

    def __len__(self):
        return self.dataset_size
