from pathlib import Path
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image

dataset_dir = Path("balanced_binary_dataset")

# Examine dataset structure
print("Dataset structure:")
for item in dataset_dir.iterdir():
    if item.is_dir():
        print(f"  {item.name}")
        # count files in each class
        for class_dir in item.iterdir():
            if class_dir.is_dir():
                num_files = sum(1 for f in class_dir.iterdir() if f.is_file() and not f.name.startswith("._"))
                print(f"    {class_dir.name}: {num_files} files")


class DICOMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir example:
            sample_dataset/train
            sample_dataset/test

        Folder structure:
            root_dir/
                AMD_yes/
                No_known_AMD/
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_names = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        self.samples = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for file_path in class_dir.iterdir():
                if file_path.is_file() and not file_path.name.startswith("._"):
                    self.samples.append((file_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        ds = pydicom.dcmread(str(file_path))
        image = ds.pixel_array.astype(np.float32)

        # normalize to 0-255
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max()
        image = (image * 255).astype(np.uint8)

        # handle image shape
        if image.ndim == 2:
            # grayscale -> RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            # if channel-first (C, H, W), convert to (H, W, C)
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.transpose(image, (1, 2, 0))

            # if already (H, W, 3), keep as is
            if image.shape[-1] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape} for file {file_path}")

        else:
            raise ValueError(f"Unsupported image shape: {image.shape} for file {file_path}")

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = DICOMDataset("full_dataset/train", transform=transform)
val_dataset = DICOMDataset("full_dataset/val", transform=transform)
test_dataset = DICOMDataset("full_dataset/test", transform=transform)

print("\nClasses:", train_dataset.class_names)
print("Class to index:", train_dataset.class_to_idx)
print("\nTrain size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))


# =========================================================
# 4. DataLoader
# =========================================================
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)
# =========================================================
# 5. Model: ResNet50
# =========================================================

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def print_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    print(title)
    header = "pred→\t" + "\t".join(class_names)
    print(header)
    for i, row in enumerate(cm):
        row_values = "\t".join(str(v) for v in row)
        print(f"true={class_names[i]}\t{row_values}")


model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)   # binary classification: AMD_yes vs No_known_AMD

model = model.to(device)


# =========================================================
# 6. Loss and Optimizer
# =========================================================
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=1e-4)

optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
], weight_decay=1e-4)

# =========================================================
# 7. Training function
# =========================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_acc, epoch_f1


# =========================================================
# 8. Evaluation function
# =========================================================
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_f1, cm, all_labels, all_preds


# =========================================================
# 9. Training loop
# =========================================================

num_epochs = 15
best_val_acc = 0.0
best_model_path = "balanced_resnet50_amd.pth"
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    train_loss, train_acc, train_f1 = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, val_f1, val_cm, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc:.4f}   | Val   F1: {val_f1:.4f}")
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    #print("Validation Confusion Matrix:")
    #print(val_cm)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")


    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc_history, marker="o", linewidth=2, label="Train Accuracy")
    plt.plot(epochs, val_acc_history, marker="s", linewidth=2, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xticks(epochs)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    accuracy_plot_path = "training_validation_accuracy.png"
    plt.savefig(accuracy_plot_path, dpi=200)
    plt.close()
    print(f"Saved accuracy plot to {accuracy_plot_path}")


# =========================================================
# 10. Final evaluation using best model
# =========================================================
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)

test_loss, test_acc, test_f1, test_cm, y_true, y_pred = evaluate(
    model, test_loader, criterion, device
)

print("\n===== Final Test Results =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("Confusion Matrix:")
print_confusion_matrix(test_cm, train_dataset.class_names, title="Confusion Matrix")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=train_dataset.class_names
))