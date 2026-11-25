
from pathlib import Path
import random
import shutil

from PIL import Image

# Folder project
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
PHOTOS = DATASET / "photos"
ANNOTATIONS = DATASET / "annotations" / "UFDD_val_bbx_gt.txt"
OUTPUT = ROOT / "yolo_dataset"

SEED = 42
random.seed(SEED)

def read_annotations(path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    records = []
    i = 0

    while i < len(lines):
        image_rel = lines[i]
        i += 1

        if i >= len(lines):
            raise ValueError("Jumlah anotasi tidak lengkap.")

        count = int(lines[i])
        i += 1

        boxes = []

        for _ in range(count):
            values = list(map(float, lines[i].split()))
            i += 1

            if len(values) < 4:
                raise ValueError(
                    f"Anotasi tidak valid: {image_rel}"
                )

            x, y, w, h = values[:4]
            boxes.append((x, y, w, h))

        records.append((image_rel, boxes))

    return records

def main():
    records = read_annotations(ANNOTATIONS)

    # Pastikan gambar dan label tersedia sebelum membuat dataset.
    valid_records = []

    for image_rel, boxes in records:
        image_path = PHOTOS / image_rel

        if not image_path.exists():
            print("Gambar tidak ditemukan:", image_path)
            continue

        with Image.open(image_path) as image:
            img_w, img_h = image.size

        valid = True

        for x, y, w, h in boxes:
            if (
                w <= 0 or h <= 0
                or x < 0 or y < 0
                or x + w > img_w
                or y + h > img_h
            ):
                valid = False
                break

        if valid:
            valid_records.append((image_rel, boxes))

    random.shuffle(valid_records)

    n = len(valid_records)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)

    splits = {
        "train": valid_records[:n_train],
        "val": valid_records[n_train:n_train + n_val],
        "test": valid_records[n_train + n_val:],
    }

    for split, items in splits.items():
        for image_rel, boxes in items:
            src = PHOTOS / image_rel
            stem = Path(image_rel).stem

            image_out = OUTPUT / "images" / split / f"{stem}.jpg"
            label_out = OUTPUT / "labels" / split / f"{stem}.txt"

            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)

            # Konversi gambar ke JPEG RGB untuk konsistensi.
            with Image.open(src) as image:
                image = image.convert("RGB")
                img_w, img_h = image.size
                image.save(image_out, quality=95)

            lines = []

            for x, y, w, h in boxes:
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                width = w / img_w
                height = h / img_h

                lines.append(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

            label_out.write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

    yaml_text = (
        f"path: {OUTPUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: face\n"
    )

    (OUTPUT / "data.yaml").write_text(yaml_text)

    print("\nKonversi selesai.")
    print("Total gambar valid:", n)
    print("Train:", len(splits["train"]))
    print("Validation:", len(splits["val"]))
    print("Test:", len(splits["test"]))
    print("Lokasi dataset:", OUTPUT)

if __name__ == "__main__":
    main()
