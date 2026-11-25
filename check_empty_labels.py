from pathlib import Path
from PIL import Image, ImageDraw

root = Path("yolo_dataset")
labels_dir = root / "labels" / "test"
images_dir = root / "images" / "test"

if not labels_dir.exists() or not images_dir.exists():
    raise FileNotFoundError(
        "Folder label atau gambar test tidak ditemukan. "
        "Pastikan kamu menjalankan skrip dari folder utama proyek."
    )

empty_labels = sorted(
    p for p in labels_dir.glob("*.txt")
    if not p.read_text().strip()
)

if not empty_labels:
    print("Tidak ada label kosong pada data test.")
    raise SystemExit

samples = empty_labels[:60]

cell_w, cell_h = 220, 170
cols = 4
rows = (len(samples) + cols - 1) // cols

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, label in enumerate(samples):
    image_path = images_dir / f"{label.stem}.jpg"

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    if not image_path.exists():
        draw.text((x + 5, y + 5), f"Missing: {label.stem}", fill="red")
        continue

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((cell_w - 10, cell_h - 30))
        sheet.paste(im, (x + 5, y + 5))

    draw.text(
        (x + 5, y + cell_h - 20),
        label.stem[:28],
        fill="black"
    )

output = Path("empty_labels_test.jpg")
sheet.save(output)

print(f"Total label kosong: {len(empty_labels)}")
print(f"Jumlah gambar dalam pratinjau: {len(samples)}")
print(f"Hasil disimpan di: {output.resolve()}")