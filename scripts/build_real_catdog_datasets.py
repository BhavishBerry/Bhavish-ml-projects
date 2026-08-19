"""Build 3 real-photo cat/dog datasets from the official Microsoft
'Kaggle Cats and Dogs' archive (kagglecatsanddogs_5340.zip).

Samples non-overlapping images per class, skips corrupt files (a few are
known to be broken in this archive), resizes to 128x128, and lays them out
in the same train/validation/{cat,dog} structure as the CNN notebook expects.
"""
import random
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image

ZIP_PATH = Path("/tmp/claude-1000/-home-bhavish-berry-PycharmProjects-ai-training-colllege-first-model/5fe3766d-e8ed-44cc-acb6-a2a7e5a0821a/scratchpad/kagglecatsanddogs.zip")
OUT_BASE = Path(__file__).resolve().parent.parent / "Bhavish" / "cnn_project"
TARGET_SIZE = (128, 128)
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 30
NEEDED_PER_CLASS_PER_DATASET = TRAIN_PER_CLASS + VAL_PER_CLASS  # 150
NUM_DATASETS = 3


def load_valid_image(zf: zipfile.ZipFile, name: str):
    try:
        data = zf.read(name)
        img = Image.open(BytesIO(data))
        img.verify()
        img = Image.open(BytesIO(data)).convert("RGB")
        img = img.resize(TARGET_SIZE)
        return img
    except Exception:
        return None


def main():
    rng = random.Random(7)
    zf = zipfile.ZipFile(ZIP_PATH)
    names = zf.namelist()
    cats = [n for n in names if n.startswith("PetImages/Cat/") and n.endswith(".jpg")]
    dogs = [n for n in names if n.startswith("PetImages/Dog/") and n.endswith(".jpg")]
    rng.shuffle(cats)
    rng.shuffle(dogs)

    total_needed = NEEDED_PER_CLASS_PER_DATASET * NUM_DATASETS  # 450 per class

    def collect_valid(pool, needed):
        valid = []
        i = 0
        while len(valid) < needed and i < len(pool):
            img = load_valid_image(zf, pool[i])
            if img is not None:
                valid.append(img)
            i += 1
        return valid

    cat_imgs = collect_valid(cats, total_needed)
    dog_imgs = collect_valid(dogs, total_needed)
    print(f"Collected {len(cat_imgs)} cat / {len(dog_imgs)} dog valid images")

    for ds_idx in range(NUM_DATASETS):
        out_dir = OUT_BASE / f"catdog_real_dataset_{ds_idx + 1}"
        cat_slice = cat_imgs[ds_idx * NEEDED_PER_CLASS_PER_DATASET:(ds_idx + 1) * NEEDED_PER_CLASS_PER_DATASET]
        dog_slice = dog_imgs[ds_idx * NEEDED_PER_CLASS_PER_DATASET:(ds_idx + 1) * NEEDED_PER_CLASS_PER_DATASET]

        for cls, imgs in [("cat", cat_slice), ("dog", dog_slice)]:
            train_dir = out_dir / "train" / cls
            val_dir = out_dir / "validation" / cls
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(imgs[:TRAIN_PER_CLASS]):
                img.save(train_dir / f"{i}.jpg", quality=85)
            for i, img in enumerate(imgs[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]):
                img.save(val_dir / f"{i}.jpg", quality=85)
        print(f"Built {out_dir}")


if __name__ == "__main__":
    main()
