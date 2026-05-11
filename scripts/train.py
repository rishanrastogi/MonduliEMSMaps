import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchgeo.models import FarSeg
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Dataset ---
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(os.path.join(self.image_dir, self.files[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.files[idx]))

        img = img.transpose(1, 2, 0).astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"].unsqueeze(0)
        else:
            img = torch.tensor(img).permute(2, 0, 1)
            mask = torch.tensor(mask).unsqueeze(0)

        return img, mask

# --- Augmentations ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(p=0.2),
    A.ShiftScaleRotate(p=0.3),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    ToTensorV2(),
])

# --- Load data ---
image_dir = r"C:\dev\Projects\EMS Tiles\patches\images"
mask_dir  = r"C:\dev\Projects\EMS Tiles\patches\masks"

full_dataset = RoadDataset(image_dir, mask_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train: {train_size} | Val: {val_size}")

# --- FarSeg Model (pretrained backbone, designed for road segmentation) ---


# Replace with:
model = FarSeg(backbone="resnet50", classes=2)

# wrap output for binary road detection
class FarSegBinary(torch.nn.Module):
    def __init__(self, farseg):
        super().__init__()
        self.farseg = farseg
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        out = self.farseg(x)  # (B, 2, H, W)
        return self.head(out)

model = FarSegBinary(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
model.to(device)

# --- Loss & Optimizer ---
loss_fn  = smp.losses.DiceLoss(mode="binary")
bce_fn   = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

best_val_loss = float("inf")

for epoch in range(80):
    # --- Train ---
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = loss_fn(preds, masks) + 0.5 * bce_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validate ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = loss_fn(preds, masks) + 0.5 * bce_fn(preds, masks)
            val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss / len(val_loader)
    scheduler.step(avg_val)

    print(f"Epoch {epoch+1}/80 — Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), r"C:\dev\Projects\EMS Tiles\road_model_best.pth")
        print(f"  ✓ Best model saved!")

print("Done!")