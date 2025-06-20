import torch
import numpy as np
import matplotlib.pyplot as plt
from train_unet_segmentation import UNet, SegmentationDataset

def visualize_segmentation(model, data_loader, device, num_samples=3):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            for i in range(images.size(0)):
                if shown >= num_samples:
                    return
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                gt_mask = masks[i, 0].cpu().numpy()
                pred_mask = (outputs[i, 0] > 0.5).float().cpu().numpy()
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img)
                axs[0].set_title('Image')
                axs[1].imshow(gt_mask, cmap='gray')
                axs[1].set_title('Ground Truth Mask')
                axs[2].imshow(pred_mask, cmap='gray')
                axs[2].set_title('Predicted Mask')
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
                shown += 1

if __name__ == "__main__":
    # User-settable parameters
    image_size = 128
    batch_size = 8
    init_features = 32
    # Paths
    test_dir = "Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/test/images"
    test_mask_dir = "segmentation_masks/masks_test"
    # Dataset and loader
    test_set = SegmentationDataset(test_dir, test_mask_dir, image_size=image_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(init_features=init_features).to(device)
    model.load_state_dict(torch.load("best_unet_mould.pth", map_location=device))
    # Visualize
    visualize_segmentation(model, test_loader, device, num_samples=10)
