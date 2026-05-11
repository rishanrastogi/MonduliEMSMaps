import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = r"C:\dev\Projects\EMS Tiles\patches"

# plot 4 random patches and their masks side by side
fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for i in range(4):
    img = np.load(f"{output_dir}\\images\\patch_{i:04d}.npy")
    mask = np.load(f"{output_dir}\\masks\\patch_{i:04d}.npy")

    # image: CHW -> HWC for display
    axes[i, 0].imshow(img.transpose(1, 2, 0))
    axes[i, 0].set_title(f"Satellite patch {i}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title(f"Mask patch {i}")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(r"C:\dev\Projects\EMS Tiles\patch_check.png")
plt.show()
print("Done!")