import os
import torch
from torch.utils.data import TensorDataset
from torchvision.io import read_image

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

def ObjectDataset(records, image_path): 
    images = []
    bboxes = []
    labels = []

    for _, row in records.iterrows():
        filename, img_w, img_h, class_name, xmin, ymin, xmax, ymax = row
        
        fullpath = os.path.join(image_path, filename)
        img = read_image(fullpath)
        
        # Normalize
        xmin = xmin / img_w
        ymin = ymin / img_h
        xmax = xmax / img_w
        ymax = ymax / img_h

        '''
        # Convert to CXCYWH
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        box_w = xmax - xmin
        box_h = ymax - ymin
        '''
        
        images.append(img)
        bboxes.append((xmin, ymin, xmax, ymax))
        labels.append(CLASS_NAMES.index(class_name))

    images = torch.stack(images).float() / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    bboxes = torch.tensor(bboxes, dtype=torch.float32)

    ds = TensorDataset(images, labels, bboxes)

    return ds
