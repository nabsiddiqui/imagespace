import os
import math
import random
import struct
import json
from PIL import Image, ImageDraw

# Configuration
NUM_IMAGES = 50000
THUMB_SIZE = 64  # Increased size for better visibility
ATLAS_SIZE = 4096  # Larger atlas for larger thumbs
OUTPUT_DIR = "../frontend/public/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Neo-Brutalist Palette
BRUTAL_COLORS = [
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 87, 51),  # Orange-Red
    (51, 255, 87),  # Bright Green
    (87, 51, 255),  # Purple
    (255, 255, 255),  # White
]


def create_dummy_image(index):
    bg_color = random.choice(BRUTAL_COLORS)
    img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), bg_color)
    draw = ImageDraw.Draw(img)

    # Thick black border
    draw.rectangle([0, 0, THUMB_SIZE - 1, THUMB_SIZE - 1], outline=(0, 0, 0), width=4)

    # Stark geometric shapes
    shape_color = (0, 0, 0)
    margin = 12
    if index % 4 == 0:
        draw.ellipse(
            [margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin], fill=shape_color
        )
    elif index % 4 == 1:
        draw.rectangle(
            [margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin], fill=shape_color
        )
    elif index % 4 == 2:
        draw.polygon(
            [
                (THUMB_SIZE // 2, margin),
                (margin, THUMB_SIZE - margin),
                (THUMB_SIZE - margin, THUMB_SIZE - margin),
            ],
            fill=shape_color,
        )
    else:
        draw.line(
            [margin, margin, THUMB_SIZE - margin, THUMB_SIZE - margin],
            fill=shape_color,
            width=8,
        )
        draw.line(
            [margin, THUMB_SIZE - margin, THUMB_SIZE - margin, margin],
            fill=shape_color,
            width=8,
        )

    return img


def main():
    print(f"🚀 Generating {NUM_IMAGES} Neo-Brutalist images...")

    images_per_row = ATLAS_SIZE // THUMB_SIZE
    images_per_atlas = images_per_row * images_per_row

    binary_data = bytearray()
    centers = [
        (random.uniform(-2000, 2000), random.uniform(-2000, 2000)) for _ in range(12)
    ]

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

        center = random.choice(centers)
        x = center[0] + random.gauss(0, 500)
        y = center[1] + random.gauss(0, 500)

        # Pack into Binary
        item_bin = struct.pack("<ffHHHH", x, y, current_atlas_idx, u, v, 0)
        binary_data.extend(item_bin)

        current_img_in_atlas += 1
        if current_img_in_atlas >= images_per_atlas or i == NUM_IMAGES - 1:
            atlas_img.save(
                os.path.join(OUTPUT_DIR, f"atlas_{current_atlas_idx}.jpg"),
                "JPEG",
                quality=90,
            )
            print(f"Saved atlas_{current_atlas_idx}.jpg")
            current_atlas_idx += 1
            current_img_in_atlas = 0
            atlas_img = Image.new("RGB", (ATLAS_SIZE, ATLAS_SIZE), (255, 255, 255))

    with open(os.path.join(OUTPUT_DIR, "data.bin"), "wb") as f:
        f.write(binary_data)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(
            {
                "count": NUM_IMAGES,
                "atlasCount": current_atlas_idx,
                "thumbSize": THUMB_SIZE,
            },
            f,
        )

    print(f"✅ Done. Binary size: {len(binary_data) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
