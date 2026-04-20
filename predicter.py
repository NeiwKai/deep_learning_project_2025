import os
import copy
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms_v2
from model import MobileNetMultiHead, ResNet18MultiHead, ResNet50MultiHead, UNetMultiHead
from dataset import ObjectDataset
from dl_utils import plot_predictions

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

# Define Transformations for Validation & Test (No Augmentation, but with Bounding Box Support)
test_transforms = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    imgs, labels, bboxes, sizes = zip(*batch)

    return (
        torch.stack(imgs),
        torch.stack(labels),
        torch.stack(bboxes),
        list(sizes)   # force to tuple of int
    )

if __name__ == '__main__':
    TESTING_CSV_FILE = 'Cloud-Classification-7/test/_annotations.csv'
    TESTING_IMAGE_DIR = 'Cloud-Classification-7/test'

    testing_image_records = pd.read_csv(TESTING_CSV_FILE)

    test_image_path = os.path.join(os.getcwd(), TESTING_IMAGE_DIR)

    test_ds = ObjectDataset(testing_image_records, test_image_path, transform=test_transforms)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, collate_fn=collate_fn) # Just want to see different result

    num_classes = len(CLASS_NAMES)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    model_base = MobileNetMultiHead(num_classes=num_classes)
    model = copy.deepcopy(model_base)
    model = model.to(device)
    model.load_state_dict(torch.load(f"{model._get_name()}_best_vloss.pth", map_location=device))

    test_images, test_labels, test_bboxes, test_og_size = next(iter(test_dl))
    test_images = test_images.to(device)

    with torch.no_grad():
        test_preds, test_bboxes_pred = model(test_images)
    test_images = test_images.cpu()

    # Plot predictions
    plot_predictions(
        test_images, test_labels, test_bboxes, CLASS_NAMES,
        test_preds.cpu(), test_bboxes_pred.cpu(), test_og_size, 
        num_samples=16, save_path="predictions.jpg",
    )
    print("Save prediction as 'predictions.jpg'!")

