from ultralytics import YOLO
import os

DATASET_PATH = '/Users/arshiasingh/Downloads/dataset'  
EPOCHS = 100
IMGSZ = 640
BATCH = 8          #

print("── Verifying dataset ──")
for split in ['train', 'valid', 'test']:
    path = os.path.join(DATASET_PATH, split, 'images')
    count = len(os.listdir(path))
    print(f'  {split}: {count} images')

model = YOLO('yolo11s.pt')  # auto-downloads on first run

results = model.train(
    data=f'{DATASET_PATH}/data.yaml',
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=20,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    device='mps',      

    degrees=0,
    flipud=0.0,
    fliplr=0.0,
    hsv_v=0.0,
    mosaic=1.0,
    mixup=0.1,
    scale=0.3,
    project='runs/microplastic',
    name='baseline_640',
    save=True,
    save_period=10,      # checkpoint every 10 epochs
)

# ── Per-class mAP ─────────────────────────────────────────
print("\n── Per-class mAP@50 ──")
metrics = model.val()
class_names = ['fiber', 'film', 'fragment', 'pallet']
for name, score in zip(class_names, metrics.box.maps):
    status = "✓" if score >= 0.5 else "⚠ needs work"
    print(f'  {name:<12} {score:.3f}  {status}')
print(f'\n  Overall mAP@50:    {metrics.box.map50:.3f}')
print(f'  Overall mAP@50-95: {metrics.box.map:.3f}')