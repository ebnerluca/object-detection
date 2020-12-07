import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


class Wrapper:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Model()

        model_dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(model_dir_path, "weights/model.pt")
        self.model.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch_or_image):
        if len(batch_or_image.shape) == 3:
            batch = [batch_or_image]
        else:
            batch = batch_or_image

        with torch.no_grad():
            preds = self.model([to_tensor(img).to(device=self.device, dtype=torch.float) for img in batch])

        boxes = []
        labels = []
        scores = []
        for pred in preds:
            boxes.append(pred["boxes"].cpu().numpy())
            labels.append(pred["labels"].cpu().numpy())
            scores.append(pred["scores"].cpu().numpy())

        return boxes, labels, scores


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)  # 5 is the number of classes

    def forward(self, x, y=None):
        """
        return prediction for input x
        """
        return self.model(x) if y is None else self.model(x, y)
