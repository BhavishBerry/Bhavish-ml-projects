"""Generate 3 synthetic cat/dog face-image datasets (64x64 PNG) for the CNN project.

No internet/photo source is used; images are procedurally drawn shapes that
are visually distinguishable as "cat-like" vs "dog-like" faces (round head +
pointy ears + whiskers vs round head + floppy ears + tongue), with randomized
colors/positions so each dataset is unique. Mirrors the train/validation
layout of the existing shapes_dataset (120 train + 30 val per class).
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw

SIZE = 64
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 30

CAT_COLORS = [(255, 178, 102), (200, 160, 120), (160, 160, 160), (90, 60, 40), (240, 220, 180)]
DOG_COLORS = [(200, 140, 90), (140, 100, 60), (230, 200, 160), (110, 80, 50), (60, 50, 45)]
BG_COLORS = [(245, 245, 245), (230, 240, 250), (250, 240, 230), (235, 235, 235)]


def draw_cat(rng: random.Random) -> Image.Image:
    bg = rng.choice(BG_COLORS)
    fur = rng.choice(CAT_COLORS)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    cx, cy = SIZE // 2 + rng.randint(-3, 3), SIZE // 2 + rng.randint(-2, 4)
    r = rng.randint(18, 22)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fur)
    ear_h = rng.randint(12, 16)
    d.polygon([(cx - r + 2, cy - r + 6), (cx - r - 4, cy - r - ear_h), (cx - r + 14, cy - r + 2)], fill=fur)
    d.polygon([(cx + r - 2, cy - r + 6), (cx + r + 4, cy - r - ear_h), (cx + r - 14, cy - r + 2)], fill=fur)
    eye_r = 2
    d.ellipse([cx - 9, cy - 3, cx - 9 + eye_r * 2, cy - 3 + eye_r * 2], fill=(20, 20, 20))
    d.ellipse([cx + 9 - eye_r * 2, cy - 3, cx + 9, cy - 3 + eye_r * 2], fill=(20, 20, 20))
    d.polygon([(cx - 2, cy + 4), (cx + 2, cy + 4), (cx, cy + 7)], fill=(180, 100, 100))
    for dx, dy in [(-1, 0), (-1, 2), (-1, 4)]:
        d.line([(cx - 12, cy + 4 + dy), (cx - 24, cy + dx + 4 + dy)], fill=(80, 80, 80))
        d.line([(cx + 12, cy + 4 + dy), (cx + 24, cy + dx + 4 + dy)], fill=(80, 80, 80))
    return img


def draw_dog(rng: random.Random) -> Image.Image:
    bg = rng.choice(BG_COLORS)
    fur = rng.choice(DOG_COLORS)
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    cx, cy = SIZE // 2 + rng.randint(-3, 3), SIZE // 2 + rng.randint(-2, 4)
    r = rng.randint(19, 23)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fur)
    ear_w, ear_h = rng.randint(8, 11), rng.randint(16, 20)
    d.ellipse([cx - r - 2, cy - r + 4, cx - r - 2 + ear_w, cy - r + 4 + ear_h], fill=fur)
    d.ellipse([cx + r + 2 - ear_w, cy - r + 4, cx + r + 2, cy - r + 4 + ear_h], fill=fur)
    eye_r = 2
    d.ellipse([cx - 9, cy - 2, cx - 9 + eye_r * 2, cy - 2 + eye_r * 2], fill=(20, 20, 20))
    d.ellipse([cx + 9 - eye_r * 2, cy - 2, cx + 9, cy - 2 + eye_r * 2], fill=(20, 20, 20))
    d.ellipse([cx - 12, cy + 6, cx + 12, cy + 16], fill=fur)
    d.ellipse([cx - 3, cy + 8, cx + 3, cy + 12], fill=(30, 30, 30))
    d.line([(cx, cy + 12), (cx, cy + 20)], fill=(220, 100, 120), width=3)
    return img


def build_dataset(out_dir: Path, seed: int):
    rng = random.Random(seed)
    specs = [
        ("train", "cat", TRAIN_PER_CLASS, draw_cat),
        ("train", "dog", TRAIN_PER_CLASS, draw_dog),
        ("validation", "cat", VAL_PER_CLASS, draw_cat),
        ("validation", "dog", VAL_PER_CLASS, draw_dog),
    ]
    for split, cls, count, fn in specs:
        d = out_dir / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            fn(rng).save(d / f"{i}.png")
    print(f"Built dataset at {out_dir} (seed={seed})")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "Bhavish" / "cnn_project"
    build_dataset(base / "catdog_dataset_1", seed=1)
    build_dataset(base / "catdog_dataset_2", seed=2)
    build_dataset(base / "catdog_dataset_3", seed=3)
