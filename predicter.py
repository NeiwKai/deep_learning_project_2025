import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import MobileNetMultiHead, ResNet18MultiHead, ResNet50MultiHead
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

if __name__ == '__main__':
    TESTING_CSV_FILE = 'Cloud-Classification-7/test/_annotations.csv'
    TESTING_IMAGE_DIR = 'Cloud-Classification-7/test'

    testing_image_records = pd.read_csv(TESTING_CSV_FILE)

    test_image_path = os.path.join(os.getcwd(), TESTING_IMAGE_DIR)

    test_ds = ObjectDataset(testing_image_records, test_image_path)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    num_classes = len(CLASS_NAMES)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    model_base = ResNet18MultiHead(num_classes=num_classes)
    model = model_base
    model = model.to(device)
    model.load_state_dict(torch.load(f"{model._get_name()}_best_vloss.pth"))

    test_images, test_labels, test_bboxes = next(iter(test_dl))
    test_images = test_images.to(device)

    with torch.no_grad():
        test_preds, test_bboxes_pred = model(test_images)
    test_images = test_images.cpu()

    # Plot predictions
    plot_predictions(
        test_images, test_labels, test_bboxes, CLASS_NAMES,
        test_preds.cpu(), test_bboxes_pred.cpu(), 
        num_samples=16, save_path="predictions.jpg",
    )

