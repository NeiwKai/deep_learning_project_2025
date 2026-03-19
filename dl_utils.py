import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler # Automatically run operations in lower precision (FP16) 

def train_one_epoch(
        dataloader, model, losses, optimizer, 
        epoch, device, writer, log_step_interval=50
    ):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    scaler = GradScaler()

    for i, (X, y, bboxes) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device):
            cl_pred, bb_pred = model(X)
            cl_loss = losses["cl_head"](cl_pred, y)
            bb_loss = losses["bb_head"](bb_pred, bboxes)
            total_loss = cl_loss + bb_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        if (i+1) % log_step_interval == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f} (Class: {cl_loss.item():.4f}, BBox: {bb_loss.item():.4f})")
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(dataloader) + i)


def test(dataloader, model, losses, device):
    num_batches = len(dataloader)
    model.eval()
    cl_loss, bb_loss = 0, 0
    y_preds, y_trues = [], []
    bbox_preds, bbox_trues = [], []

    with torch.no_grad():
        for i, (X, y, bboxes) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)
            bboxes = bboxes.to(device)

            cl_pred, bb_pred = model(X)

            cl_loss += losses["cl_head"](cl_pred, y)
            bb_loss += losses["bb_head"](bb_pred, bboxes)

            y_preds.append(cl_pred.argmax(1))
            y_trues.append(y)
            bbox_preds.append(bb_pred)
            bbox_trues.append(bboxes)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)
    bbox_preds = torch.cat(bbox_preds)
    bbox_trues = torch.cat(bbox_trues)

    cl_loss /= num_batches
    bb_loss /= num_batches
    return cl_loss, bb_loss, y_preds, y_trues, bbox_preds, bbox_trues


def plot_predictions(
        images, labels, bboxes_true, class_names, 
        preds=None, bboxes_pred=None, num_samples=6, 
        save_path="predictions.jpg"
    ):
    num_samples = min(num_samples, images.shape[0])  # Ensure we don't exceed batch size

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(20, 10))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_samples):
        img = images[i].detach()
        label = labels[i].item()
        bbox_t = bboxes_true[i].detach().numpy()  # Ground truth bbox

        # Unnormalize image
        img = img * std + mean
        img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

        ax = axes[i]
        ax.imshow(img)
        
        # Draw ground truth bbox
        img_h, img_w, _ = img.shape
        # print(img.shape, img_w, img_h)
        _draw_bbox(ax, bbox_t, img_h, img_w, "green", label=f"GT: {class_names[label]}")
        
        # Draw predicted bbox
        if preds is not None:
            pred = preds[i].argmax().item() # Add argmax() to get exactly 1 label outcome
            bbox_p = bboxes_pred[i].detach().numpy()  # Predicted bbox
            _draw_bbox(ax, bbox_p, img_h, img_w, "red", label=f"Pred: {class_names[pred]}")

        ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)
    plt.close('all')


def _draw_bbox(ax, bbox, img_h, img_w, color, label=""):
    x, y, w, h = bbox
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h

    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y - 2, label, color=color, fontsize=10, bbox=dict(facecolor="white", alpha=0.6))
