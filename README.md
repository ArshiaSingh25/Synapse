# Microplastic Morphology Classifier
### Synapse Hackathon — Problem Statement 3
**Theme: Marine Ecosystem Protection & Water Quality Monitoring**

---

## 1. Dataset & Preprocessing

### Dataset
- **Source:** [Microplastic 100 — Roboflow Universe](https://universe.roboflow.com/kueranan/microplastic_100)
- **Total images:** 960 (after augmentation)
- **Original images:** 400 microscope images
- **Classes:** 4 — `fiber`, `film`, `fragment`, `pallet`
- **Annotation type:** Bounding boxes (object detection)
- **Class distribution:**

| Class | Instances |
|---|---|
| fiber | 1,155 |
| film | 1,207 |
| fragment | 1,198 |
| pallet | 1,144 |

The dataset is perfectly balanced across all four morphological classes.

### Train / Val / Test Split

| Split | Images |
|---|---|
| Train | 840 |
| Validation | 60 |
| Test | 60 |

### Preprocessing (applied via Roboflow)
1. **Auto-Orient** — corrects EXIF rotation inconsistencies across microscope image exports
2. **Resize** — all images standardised to 640×640 pixels
3. **Adaptive Equalization (CLAHE)** — Contrast Limited Adaptive Histogram Equalization applied to normalise uneven microscope illumination (bright center, darker edges in circular field of view). Uses local contrast normalisation with clip limit 2.0 and tile grid 8×8, preserving inter-particle texture differences critical for morphology classification

### Augmentation (applied via Roboflow at export)
- Horizontal flip
- Vertical flip
- Rotation ±15°
- Brightness adjustment ±20%

> Blur augmentation was deliberately excluded — microscope images are always in focus by design, so blur would introduce unrealistic artifacts and potentially erase faint film-type particles.

### Additional Training-time Augmentation (YOLOv11 config)
- **Mosaic augmentation** (p=1.0) — pastes 4 images together per batch, critical for improving small-object (pallet) detection
- **Mixup** (p=0.1)
- **Random scale** (±30%) — improves detection of small pallets at variable sizes

---

## 2. Model & Performance Metrics

### Model Architecture
- **Model:** YOLOv11s (You Only Look Once, version 11 — small variant)
- **Pretrained weights:** COCO dataset (118,000 images, 80 classes)
- **Fine-tuned on:** Microplastic morphology dataset (4 classes)
- **Input resolution:** 640×640
- **Framework:** Ultralytics

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 35  |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 0.001 (cosine decay) |
| Early stopping patience | 20 epochs |
| Device | Apple MPS (M-series) |

### Performance Metrics (Validation Set — 60 images)

| Class | mAP@50 | Precision | Recall |
|---|---|---|---|
| fiber | 0.993 | 0.992 | 0.994 |
| film | 0.948 | 0.964 | 0.956 |
| fragment | 0.939 | 0.923 | 0.976 |
| pallet | 0.848 | 0.855 | 0.883 |
| **Overall** | **0.932** | **0.934** | **0.952** |

- **Overall mAP@50:** 0.932
- **Overall mAP@50-95:** 0.476
- **F1 Score:** 0.943 (at optimal confidence threshold 0.381)
- **Inference speed:** 212ms per image (CPU), 0.7ms preprocess, 0.3ms postprocess

### Confusion Matrix Highlights
- Fiber correctly classified 99% of the time — near-perfect
- Fragment correctly classified 98% of the time
- Film correctly classified 96% of the time
- Pallet correctly classified 92% of the time
- Minimal cross-class confusion — morphological features are visually distinct to the model

---

## 3. Key Features

- **Image input** — accepts `.jpg` / `.png` microscope or high-magnification smartphone images
- **Morphology classification** — detects and classifies all particles in a single image into fiber, film, fragment, or pallet (including multiple particles per image)
- **Size estimation** — computes Feret diameter (longest diagonal of bounding box in pixels), converted to micrometers (µm) using microscope scale calibration
- **Marine Ecological Threat Index (ETI)** — scientifically grounded risk score (0–100) per particle, combining:
  - Morphology score (literature-backed marine ingestion evidence)
  - Non-linear size score (zooplankton ingestion threshold at 100µm, Cambridge Marine Toxicology 2023)
  - Trophic amplification score (food web bioaccumulation multiplier for particles <100µm)
- **Pellet / Microbead detection** — fourth morphological class (pallet) fully implemented
- **Grad-CAM heatmap overlay** — per-particle explainability heatmap showing which image regions drove the classification decision, addressing the "black box" concern for regulatory use
- **Batch processing** — multiple image upload with aggregate summary report

### Novel Scientific Contribution
- **Pollution Source Attribution via Morphological Fingerprinting** — the first image-based system to identify the likely pollution source (textile effluent, fishing industry, packaging waste, personal care products, urban stormwater) from morphology distribution alone, without requiring FTIR chemical analysis. Uses confidence-weighted Morphology Distribution Vector (MDV) and cosine similarity against literature-derived source profiles.

---

## 4. Solution Architecture

```
Input Image
    ↓
Preprocessing (CLAHE → resize to 640×640)
    ↓
YOLOv11s Detection
    ↓ ↓
Bounding boxes + class + confidence
    ↓                          ↓
Size Estimation            Grad-CAM Heatmap
(Feret diameter px → µm)   (per-particle overlay)
    ↓
Marine ETI Scoring
(morphology + size + trophic amplification)
    ↓
Pollution Source Attribution
(cosine similarity → MDV fingerprint)
    ↓
Frontend Dashboard
(annotated image · per-particle table · ETI gauge · source report)


```

### Research Gap Addressed
All established ecological risk indices (PLI, PERI, PHI, RQ) require polymer chemical identification via FTIR spectroscopy. Our ETI and pollution source attribution system is the first morphology-and-size-derived risk framework computable directly from a microscope image, enabling real-time field assessment without chemical analysis.

---

## 5. References

1. Cambridge Prisms: Microplastics — Zooplankton ingestion study, 2023
2. ACS Environmental Science & Technology — Selective microplastic ingestion by copepod species, 2020
3. Frontiers in Marine Science — Biofouling and morphology effects on MP fate, 2022
4. PMC Aquatic Ecosystems — Marine food web trophic transfer of MPs, 2024
5. USGS Strategic Science Vision — Microplastic source attribution gaps, 2023

---

*Dataset: [Roboflow Universe — microplastic_100 by kueranan](https://universe.roboflow.com/kueranan/microplastic_100)*
*Model weights: YOLOv11s fine-tuned, available as `best.pt`*
