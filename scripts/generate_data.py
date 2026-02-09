import os
import math
import random
import struct
import json
from PIL import Image, ImageDraw

# Configuration
NUM_IMAGES = 50000
THUMB_SIZE = 64
ATLAS_SIZE = 4096
OUTPUT_DIR = "../frontend-pixi/public/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Neo-Brutalist Palette
BRUTAL_COLORS = [
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 87, 51),   # Orange-Red
    (51, 255, 87),   # Bright Green
    (87, 51, 255),   # Purple
    (255, 255, 255), # White
]

NUM_CLUSTERS = 12


def create_dummy_image(index):
    bg_color = random.choice(BRUTAL_COLORS)
    img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), bg_color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, THUMB_SIZE - 1, THUMB_SIZE - 1], outline=(0, 0, 0), width=4)
    shape_color = (0, 0, 0)
    margin = 12
    if index % 4 == 0:
        draw.ellipse([margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin], fill=shape_color)
    elif index % 4 == 1:
        draw.rectangle([margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin], fill=shape_color)
    elif index % 4 == 2:
        draw.polygon([(THUMB_SIZE // 2, margin), (margin, THUMB_SIZE - margin), (THUMB_SIZE - margin, THUMB_SIZE - margin)], fill=shape_color)
    else:
        draw.line([margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin], fill=shape_color, width=8)
        draw.line([margin, THUMB_SIZE - margin, THUMB_SIZE - margin, margin], fill=shape_color, width=8)
    return img


def main():
    print(f"Generating {NUM_IMAGES} test images with dual coordinates...")

    images_per_row = ATLAS_SIZE // THUMB_SIZE
    images_per_atlas = images_per_row * images_per_row

    # Generate cluster centers for UMAP and t-SNE (different layouts)
    umap_centers = [(random.uniform(-2000, 2000), random.uniform(-2000, 2000)) for _ in range(NUM_CLUSTERS)]
    tsne_centers = [(random.uniform(-2000, 2000), random.uniform(-2000, 2000)) for _ in range(NUM_CLUSTERS)]

    binary_data = bytearray()
    current_atlas_idx = 0
    current_img_in_atlas = 0
    atlas_img = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE), (255, 255, 255))

    for i in range(NUM_IMAGES):
        if i % 5000 == 0:
            print(f"Processing {i}...")

        img = create_dummy_image(i)
        col = current_img_in_atlas % images_per_row
        row = current_img_in_atlas // images_per_row
        u, v = col * THUMB_SIZE, row * THUMB_SIZE
        atlas_img.paste(img, (u, v))

        # Assign to cluster
        cluster_id = i % NUM_CLUSTERS

        # UMAP coordinates (clustered around umap_centers)
        uc = umap_centers[cluster_id]
        umap_x = uc[0] + random.gauss(0, 500)
        umap_y = uc[1] + random.gauss(0, 500)

        # t-SNE coordinates (different layout, tighter clusters)
        tc = tsne_centers[cluster_id]
        tsne_x = tc[0] + random.gauss(0, 300)
        tsne_y = tc[1] + random.gauss(0, 300)

        # Pack 24 bytes: float32 umapX, umapY, tsneX, tsneY, uint16 atlas, u, v, cluster
        item_bin = struct.pack("<ffffHHHH", umap_x, umap_y, tsne_x, tsne_y, current_atlas_idx, u, v, cluster_id)
        binary_data.extend(item_bin)

        current_img_in_atlas += 1
        if current_img_in_atlas >= images_per_atlas or i == NUM_IMAGES - 1:
            atlas_img.save(os.path.join(OUTPUT_DIR, f"atlas_{current_atlas_idx}.jpg"), "JPEG", quality=90)
            print(f"Saved atlas_{current_atlas_idx}.jpg")
            current_atlas_idx += 1
            current_img_in_atlas = 0
            atlas_img = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE), (255, 255, 255))

    with open(os.path.join(OUTPUT_DIR, "data.bin"), "wb") as f:
        f.write(binary_data)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "count": NUM_IMAGES,
            "atlasCount": current_atlas_idx,
            "thumbSize": THUMB_SIZE,
            "bytesPerImage": 24,
            "version": 2,
        }, f, indent=2)

    print(f"Done. Binary size: {len(binary_data) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
