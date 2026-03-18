import os
import pandas as pd
from datetime import datetime

import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as transforms_v2
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score
)
import torch.nn.functional as F

from dataset import ObjectDataset
from model import MultiTaskModel
from dl_utils import train_one_epoch, test

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
num_classes = len(CLASS_NAMES)
IMG_SIZE = (224, 224)

# Train set
TRAINING_CSV_FILE = 'Cloud-Classification-7/train/_annotations.csv'
TRAINING_IMAGE_DIR = 'Cloud-Classification-7/train'

training_image_records = pd.read_csv(TRAINING_CSV_FILE)

train_image_path = os.path.join(os.getcwd(), TRAINING_IMAGE_DIR)

train_ds = ObjectDataset(training_image_records, train_image_path)

# Validate set
VALIDATING_CSV_FILE = 'Cloud-Classification-7/valid/_annotations.csv'
VALIDATING_IMAGE_DIR = 'Cloud-Classification-7/valid'

validating_image_records = pd.read_csv(VALIDATING_CSV_FILE)

valid_image_path = os.path.join(os.getcwd(), VALIDATING_IMAGE_DIR)

valid_ds = ObjectDataset(validating_image_records, valid_image_path)

# Testing set
TESTING_CSV_FILE = 'Cloud-Classification-7/test/_annotations.csv'
TESTING_IMAGE_DIR = 'Cloud-Classification-7/test'

testing_image_records = pd.read_csv(TESTING_CSV_FILE)

test_image_path = os.path.join(os.getcwd(), TESTING_IMAGE_DIR)

test_ds = ObjectDataset(testing_image_records, test_image_path)



train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

losses = {
    "cl_head": nn.CrossEntropyLoss(),  # Loss for class prediction
    "bb_head": nn.MSELoss() # Loss for bounding box prediction
}

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# Model Initialize
model_base = MultiTaskModel(num_classes=num_classes)
model = model_base
model = model.to(device)


learning_rate = 1e-5
batch_size = 4   
epochs = 10           
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

best_vloss = 100000.
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")
    train_one_epoch(train_dl, model, losses, optimizer, epoch, device, writer, log_step_interval=1)

    # Compute train & validation loss
    train_loss, train_bbox_loss, train_y_preds, train_y_trues, train_bbox_preds, train_bbox_trues = test(
        train_dl, model, losses, device
    )
    val_loss, val_bbox_loss, val_y_preds, val_y_trues, val_bbox_preds, val_bbox_trues = test(
        valid_dl, model, losses, device
    )

    # Compute classification metrics
    train_accuracy = multiclass_accuracy(train_y_preds, train_y_trues).item()
    train_f1 = multiclass_f1_score(train_y_preds, train_y_trues).item()
    val_accuracy = multiclass_accuracy(val_y_preds, val_y_trues).item()
    val_f1 = multiclass_f1_score(val_y_preds, val_y_trues).item()

    # Compute bounding box MSE
    train_bbox_mse = F.mse_loss(train_bbox_preds, train_bbox_trues).item()
    val_bbox_mse = F.mse_loss(val_bbox_preds, val_bbox_trues).item()

    # Log training performance
    writer.add_scalars('Train vs. Valid/loss', 
        {'train': train_loss, 'valid': val_loss}, 
        epoch)
    writer.add_scalars('Train vs. Valid/bbox_mse', 
        {'train': train_bbox_mse, 'valid': val_bbox_mse}, 
        epoch)
    writer.add_scalars('Train vs. Valid/acc', 
        {'train': train_accuracy, 'valid': val_accuracy}, 
        epoch)
    writer.add_scalars('Train vs. Valid/f1', 
        {'train': train_f1, 'valid': val_f1}, 
        epoch)

    # Save the best model
    if val_loss < best_vloss:
        best_vloss = val_loss
        torch.save(model.state_dict(), f'{model._get_name()}_best_vloss.pth')
        print(f'Saved best model to {model._get_name()}_best_vloss.pth')

print("Training Complete!")


# -- Testing --

model_best = model_base
model_best = model_best.to(device)
model_best.load_state_dict(torch.load(f"{model._get_name()}_best_vloss.pth"))

# Evaluate on the test set
test_loss, test_bbox_loss, test_y_preds, test_y_trues, test_bbox_preds, test_bbox_trues = test(
    test_dl, model, losses, device
)

# Compute test classification metrics
test_accuracy = multiclass_accuracy(test_y_preds, test_y_trues).item()
test_f1 = multiclass_f1_score(test_y_preds, test_y_trues).item()

# Compute bounding box MSE
test_bbox_mse = F.mse_loss(test_bbox_preds, test_bbox_trues).item()

print(f"\nTest Results:")
print(f"Classification Loss: {test_loss:.4f}")
print(f"Bounding Box MSE: {test_bbox_mse:.4f}")
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"F1 Score: {test_f1:.2f}")
