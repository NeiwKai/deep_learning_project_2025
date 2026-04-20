import os
import torch
from PIL import Image
from torch.utils.data import TensorDataset, Dataset
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage

CLASS_NAMES = [
    "altocumulus",
    "altostratus",
    "cirrocumulus",
    "cirrostratus",
    "cirrus",
    "cumulonimbus",
    "cumulus",
    "nimbostratus",
    "stratocumulus",
    "stratus",
]

class ObjectDataset(Dataset):
    def __init__(self, records, image_dir, transform=None):
        self.records = records.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        filename, img_w, img_h, class_name, xmin, ymin, xmax, ymax = row

        fullpath = os.path.join(self.image_dir, filename)
        img = Image.open(fullpath).convert("RGB")

        # normalize (0–1)
        bbox = [
            xmin / img_w,
            ymin / img_h,
            xmax / img_w,
            ymax / img_h
        ]

        label = CLASS_NAMES.index(class_name)

        # convert to absolute for transforms
        w, h = img.size
        bbox_abs = BoundingBoxes(
            [bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h],
            format="xyxy",
            canvas_size=(h, w)
        )

        img = TVImage(img)

        if self.transform:
            img, bbox_abs = self.transform(img, bbox_abs)

        # extract transformed bbox
        bbox_abs = bbox_abs[0]

        # convert back to normalized xyxy
        bbox_norm = torch.tensor([
            bbox_abs[0] / w,
            bbox_abs[1] / h,
            bbox_abs[2] / w,
            bbox_abs[3] / h
        ], dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long), bbox_norm, (int(w), int(h)) # last 2 is og_size of image


