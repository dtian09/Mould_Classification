import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features)
        self.encoder2 = UNet._block(features, features * 2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = UNet._block(features * 8, features * 16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = UNet._block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = UNet._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = UNet._block(features * 2, features)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

# --- Segmentation Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        self.transform_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        mask = (mask > 0).float()  # binary mask
        return image, mask

# --- Training Loop ---
def train_unet(model, train_loader, val_loader, device, num_epochs=30, lr=1e-3, patience=5):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        train_loss = round(train_loss, 2)
        val_loss = round(val_loss, 2)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.2f}, Val Loss={val_loss:.2f}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_unet_mould.pth")
            print("Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    if union == 0:
        return float('nan')
    return intersection / union

def compute_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum().item()
    return 2. * intersection / (pred.sum().item() + target.sum().item() + 1e-8)

def evaluate_unet(model, test_loader, device):
    model.eval()
    iou_scores = []
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                pred = outputs[i, 0]
                target = masks[i, 0]
                iou = compute_iou(pred, target)
                dice = compute_dice(pred, target)
                iou_scores.append(iou)
                dice_scores.append(dice)
    mean_iou = float(np.nanmean(iou_scores))
    mean_dice = float(np.nanmean(dice_scores))
    mean_iou = round(mean_iou, 2)
    mean_dice = round(mean_dice, 2)
    print(f"Test Mean IoU: {mean_iou:.2f}, Mean Dice: {mean_dice:.2f}")
    wandb.log({"test_mean_iou": mean_iou, "test_mean_dice": mean_dice})
    return mean_iou, mean_dice

if __name__ == "__main__":
    # User-settable hyperparameters
    init_features = 32
    image_size = 128
    batch_size = 8
    num_epochs = 30
    lr = 1e-3
    patience = 5

    wandb.init(
        project="unet-mould-segmentation",
        entity="dtian",
        config={
            "init_features": init_features,
            "image_size": image_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "patience": patience
        }
    )
    image_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/images"
    mask_dir = "segmentation_masks/masks_train"
    train_set = SegmentationDataset(
        image_dir,
        mask_dir,
        image_size=image_size
    )
    val_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/valid/images"
    val_mask_dir = "segmentation_masks/masks_valid"  
    val_set = SegmentationDataset(val_dir, val_mask_dir, image_size=image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(init_features=init_features).to(device)
    train_unet(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=lr, patience=patience)
    print("Training complete. Best model saved as best_unet_mould.pth")
    # Test set evaluation
    test_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/test/images"
    test_mask_dir = "segmentation_masks/masks_test"  
    test_set = SegmentationDataset(test_dir, test_mask_dir, image_size=image_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Load best model
    model.load_state_dict(torch.load("best_unet_mould.pth", map_location=device))
    evaluate_unet(model, test_loader, device)
