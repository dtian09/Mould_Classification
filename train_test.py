'''
input: an imbalanced training set: 
            Class 0: 108 samples
            Class 1: 17 samples
            Class 2: 64 samples
            Class 3: 546 samples

pipeline: training set -> oversample -> train with early stopping on validation performance -> test

performance score: average TPR across all classes
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import wandb
from tqdm import tqdm
from vit import ViT
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, precision_score, f1_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.best_val_tpr = 0.0
        self.patience_counter = 0

    def compute_avg_tpr(self, all_labels, all_preds, n_classes):
        tprs = []
        for i in range(n_classes):
            tp = sum((all_labels == i) & (all_preds == i))
            fn = sum((all_labels == i) & (all_preds != i))
            denom = tp + fn
            tpr = tp / denom if denom > 0 else 0.0
            tprs.append(tpr)
        return float(np.mean(tprs)), tprs

    def train_epoch(self, n_classes):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        train_loss /= total
        train_acc = correct / total
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        avg_tpr, tprs = self.compute_avg_tpr(all_labels, all_preds, n_classes)
        return train_loss, train_acc, avg_tpr, tprs

    def train(self, n_classes=4):
        for epoch in range(self.num_epochs):
            train_loss, train_acc, train_avg_tpr, train_tprs = self.train_epoch(n_classes)
            train_loss = round(train_loss, 2)
            train_acc = round(train_acc, 2)
            train_avg_tpr = round(train_avg_tpr, 4)
            val_loss, val_acc, val_avg_tpr, val_tprs = self.validate(n_classes)
            val_loss = round(val_loss, 2)
            val_acc = round(val_acc, 2)
            val_avg_tpr = round(val_avg_tpr, 4)
            print(f"Epoch {epoch+1}: Train Loss={train_loss}, Train Acc={train_acc}, Train Avg TPR={train_avg_tpr}, Val Loss={val_loss}, Val Acc={val_acc}, Val Avg TPR={val_avg_tpr}")
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_avg_tpr": train_avg_tpr,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_avg_tpr": val_avg_tpr
            })
            # Early stopping based on average TPR
            if val_avg_tpr > self.best_val_tpr:
                self.best_val_tpr = val_avg_tpr
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_vit_mould.pth')
            else:
                self.patience_counter += 1
                if self.best_val_tpr >= 0.8 and self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # Load best model
        self.model.load_state_dict(torch.load('best_vit_mould.pth'))

    def validate(self, n_classes):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        val_loss /= total
        val_acc = correct / total
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        avg_tpr, tprs = self.compute_avg_tpr(all_labels, all_preds, n_classes)
        return val_loss, val_acc, avg_tpr, tprs

    
def test(model, test_loader, device, n_classes=4):
    model.eval()
    correct = 0
    total = 0
    # For TPR calculation
    true_positives = [0] * n_classes
    false_negatives = [0] * n_classes
    total_per_class = [0] * n_classes
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                total_per_class[label] += 1
                if label == pred:
                    true_positives[label] += 1
                else:
                    false_negatives[label] += 1
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    test_acc = correct / total
    test_acc = round(test_acc, 2)
    print(f"Test Accuracy: {test_acc}")
    wandb.log({"test_acc": test_acc})
    # Compute and print TPR for each class
    print("TPR (Recall) per class:")
    for i in range(n_classes):
        denom = true_positives[i] + false_negatives[i]
        tpr = true_positives[i] / denom if denom > 0 else 0.0
        print(f"Class {i}: TPR = {tpr:.2f} ({true_positives[i]}/{denom})")
        wandb.log({f"TPR_class_{i}": tpr})
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    print("Confusion Matrix:")
    print(cm)
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds, class_names=[str(i) for i in range(n_classes)])})
    # Precision and F1-score
    precision = precision_score(all_labels, all_preds, labels=list(range(n_classes)), average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, labels=list(range(n_classes)), average=None, zero_division=0)
    for i in range(n_classes):
        print(f"Class {i}: Precision = {precision[i]:.2f}, F1-score = {f1[i]:.2f}")
        wandb.log({f"Precision_class_{i}": precision[i], f"F1_class_{i}": f1[i]})
    return test_acc

class MouldDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                fname, label = line.strip().split('\t')
                if label.isdigit():
                    self.samples.append((fname, int(label)))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        # Remove .txt extension and use corresponding image extension
        img_base = fname[:-4]
        # Try all possible image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_path = os.path.join(self.image_dir, img_base + ext)
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(f"Image for {fname} not found.")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64 # number of images in a batch
    n_layers = 6  # Number of transformer encoder layers
    n_heads = 4 # number of attention heads
    num_epochs = 50  # Max epochs
    patience = 5  # Early stopping patience
    lr = 3e-4  # Learning rate

    # Initialize wandb
    wandb.init(project="vit-mould", 
            entity="dtian",
            config={
                "batch_size": batch_size,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "num_epochs": num_epochs,
                "patience": patience,
                "lr": lr
            })

    # Instantiate model and count parameters
    model = ViT(img_size=128, patch_size=16, n_layers=n_layers, n_heads=n_heads, n_classes=4, in_channels=3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params}")

    # Data preparation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Use oversampled labels for training set
    oversampled_train_set = MouldDataset(
        image_dir="Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/images",
        label_file="train_mould_size_labels_oversampled.txt",
        transform=transform
    )
    # Use original labels for evaluation
    original_train_set = MouldDataset(
        image_dir="Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/train/images",
        label_file="train_mould_size_labels.txt",
        transform=transform
    )
    val_set = MouldDataset(
        image_dir="Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/valid/images",
        label_file="valid_mould_size_labels.txt",
        transform=transform
    )
    test_set = MouldDataset(
        image_dir="Mould detection single label.v12-phase-1-yolov11.yolov7pytorch/test/images",
        label_file="test_mould_size_labels.txt",
        transform=transform
    )

    train_loader = DataLoader(oversampled_train_set, batch_size=batch_size, shuffle=True)
    original_train_loader = DataLoader(original_train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training and testing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience)
    trainer.train(n_classes=4)
    print("\nEvaluating on original training set:")
    test(model, original_train_loader, device, n_classes=4)
    print("\nEvaluating on test set:")
    test(model, test_loader, device, n_classes=4)