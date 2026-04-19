import copy
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms_v2
from model import MobileNetMultiHead, ResNet18MultiHead, ResNet50MultiHead, UNetMultiHead
from dataset import ObjectDataset
import matplotlib.pyplot as plt
import streamlit as st


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

def load_model():
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
    model.load_state_dict(torch.load(f"{model._get_name()}_best_vloss.pth"))

    return model, device

def _draw_bbox(ax, bbox, img_w, img_h, color, label=""):
    x1, y1, x2, y2 = bbox

    # Scale the bounding box to fit with original image (not resized one)
    scale_x = img_w / 224
    scale_y = img_h / 224

    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y


    x = x1 * img_w
    y = y1 * img_h
    w = (x2 - x1) * img_w
    h = (y2 - y1) * img_h

    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    tx, ty = x, max(0, y - 5)

    ax.text(
        tx, ty, label, color=color, fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7),
        ha="left", va="top"
    )

def process_image(img, cl_pred=None, bb_pred=None, class_names=None):
    if cl_pred is not None and bb_pred is not None:
        fig, axes = plt.subplots(figsize=(6, 6))

        ax = axes
        ax.imshow(img)

        # Draw pred bbox
        img_w, img_h = img.size
        print(img_w, img_h)
        cl_pred = cl_pred.argmax().item()
        bb_pred = bb_pred.detach().cpu().numpy().reshape(-1)
        _draw_bbox(ax, bb_pred, img_w, img_h, "red", label=f"Pred: {class_names[cl_pred]}")
        ax.axis("off")

        return fig

    

if __name__ == '__main__':
    num_classes = len(CLASS_NAMES)

    st.title("Cloud Classification")
    st.write("Upload an cloud image to classify")

    uploaded_file = st.file_uploader("Upload cloud image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded cloud image", width='stretch')

        try:
            model, device = load_model()

            x = test_transforms(image)
            x = x.unsqueeze(0)
            x = x.to(device)

            st.write(f"Input shape: {tuple(x.shape)}")

            with torch.no_grad():
                model.eval()
                cl_pred, bb_pred = model(x)
            x = x.cpu()
            fig = process_image(image, cl_pred, bb_pred, class_names=CLASS_NAMES)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during processing: {e}")
