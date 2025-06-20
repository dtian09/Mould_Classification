"""
Combine ViT classification (mould area class) and U-Net segmentation (predicted mask)
using a Model Contextual Protocol (MCP).

MCP Example: Use ViT's predicted class to contextualize or filter U-Net's mask.
- If ViT predicts '0' (normal), suppress all predicted mould in the mask.
- If ViT predicts '1', '2', or '3', keep U-Net's mask as is.
"""

import torch
import numpy as np
from PIL import Image
from train_unet_segmentation import UNet, SegmentationDataset
from vit import ViT  # assumes you have a ViT model class
from torchvision import transforms
import matplotlib.pyplot as plt

def mcp_combine(vit_class, unet_mask):
    """
    vit_class: int, predicted class from ViT (0-3)
    unet_mask: numpy array, predicted mask from U-Net (0=normal, 1=mould)
    Returns: numpy array, MCP-combined mask
    """
    if vit_class == 0:
        # If ViT says normal, suppress all mould in mask
        return np.zeros_like(unet_mask)
    else:
        # Otherwise, keep U-Net's prediction
        return unet_mask

def load_vit_model(weights_path, device):
    model = ViT(n_classes=4)  # adjust as needed
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_unet_model(weights_path, device, init_features=32):
    model = UNet(init_features=init_features)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # Paths and parameters
    # Use a training image as input
    image_path = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/images/_-2-_png_jpg.rf.1a5236a0de954b10e6af5959920dcd80.jpg"
    unet_weights = "best_unet_mould.pth"
    vit_weights = "best_vit_mould.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128

    # Class labels
    class_labels = {
        0: "Normal (No Mould)",
        1: "Small/Medium Mould",
        2: "Large Mould",
        3: "Extra Large Mould"
    }

    # Load models
    vit_model = load_vit_model(vit_weights, device)
    unet_model = load_unet_model(unet_weights, device)

    # Preprocess image for ViT
    vit_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_vit = vit_transform(img).unsqueeze(0).to(device)

    # ViT prediction
    with torch.no_grad():
        vit_logits = vit_model(img_vit)
        vit_class = vit_logits.argmax(dim=1).item()

    # Preprocess image for U-Net
    unet_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_unet = unet_transform(img).unsqueeze(0).to(device)

    # U-Net prediction
    with torch.no_grad():
        unet_pred = unet_model(img_unet)
        unet_mask = (unet_pred[0, 0] > 0.5).float().cpu().numpy()

    # MCP combination
    combined_mask = mcp_combine(vit_class, unet_mask)

    # Visualization: show input image, combined mask, and class label
    img_disp = np.array(img.resize((image_size, image_size)))
    mask_disp = (combined_mask * 255).astype(np.uint8)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img_disp)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    axs[1].imshow(mask_disp, cmap='gray')
    axs[1].set_title("U-Net Predicted Mask")
    axs[1].axis('off')
    # Add ViT class label at the bottom of the mask subplot
    axs[1].text(
        0.5, -0.12, f"ViT: {class_labels[vit_class]}",
        fontsize=12, color='black', ha='center', va='top', transform=axs[1].transAxes
    )
    plt.tight_layout()
    plt.savefig("combined_mask_with_label.png")
    plt.show()

    print(f"ViT predicted class: {vit_class} ({class_labels[vit_class]})")
    print("Combined mask with label saved as combined_mask_with_label.png")

if __name__ == "__main__":
    main()