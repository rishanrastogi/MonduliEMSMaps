import rasterio
import numpy as np
import os
from rasterio.windows import Window

# --- Paths ---
satellite_path = r"C:\dev\Projects\EMS Tiles\Square2_32737.tif"
mask_path = r"C:\dev\Projects\EMS Tiles\square_02v2_mask.tif"
output_dir = r"C:\dev\Projects\EMS Tiles\patches"

os.makedirs(output_dir + r"\images", exist_ok=True)
os.makedirs(output_dir + r"\masks", exist_ok=True)

PATCH_SIZE = 256
MIN_ROAD_PIXELS = 50  # skip patches with almost no roads

with rasterio.open(satellite_path) as sat, rasterio.open(mask_path) as msk:
    H, W = msk.height, msk.width
    count = 0

    for y in range(0, H - PATCH_SIZE, PATCH_SIZE):
        for x in range(0, W - PATCH_SIZE, PATCH_SIZE):
            window = Window(x, y, PATCH_SIZE, PATCH_SIZE)

            img = sat.read([1, 2, 3], window=window)   # RGB
            mask = msk.read(1, window=window)           # grayscale

            # only save patches that contain roads
            if np.sum(mask > 0) < MIN_ROAD_PIXELS:
                continue

            np.save(f"{output_dir}\\images\\patch_{count:04d}.npy", img)
            np.save(f"{output_dir}\\masks\\patch_{count:04d}.npy", mask)
            count += 1

print(f"Saved {count} patches with roads")