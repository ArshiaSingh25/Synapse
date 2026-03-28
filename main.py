from ultralytics import YOLO

model = YOLO('/Users/arshiasingh/Downloads/dataset/runs/detect/runs/microplastic/baseline_640/weights/best.pt')

# ── Step 1: Evaluate on val set (tuning/debugging) ────────
metrics = model.val(
    split='val',
    plots=True,
    project='runs/microplastic',
    name='val_eval'
)

class_names = ['fiber', 'film', 'fragment', 'pallet']
print("\n── Per-class mAP@50 (val set) ──")
for name, score in zip(class_names, metrics.box.ap50):  # ap50 not maps
    status = "✓" if score >= 0.5 else "⚠"
    print(f"  {name:<12} {score:.3f}  {status}")

print(f"\n  Overall mAP@50:    {metrics.box.map50:.3f}")
print(f"  Overall mAP@50-95: {metrics.box.map:.3f}")

# ── Step 2: Predict on test set (final results for judges) ─
model.predict(
    source='/Users/arshiasingh/Downloads/dataset/test/images',
    save=True,
    save_conf=True,
    conf=0.25,
    project='runs/microplastic',
    name='test_predictions'
)

print("\nDone. Check:")
print("  runs/microplastic/val_eval/        → confusion matrix, curves")
print("  runs/microplastic/test_predictions/ → annotated test images")