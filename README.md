# Microplastic Morphology Classifier
### Synapse Hackathon — Problem Statement 3
**Theme: Marine Ecosystem Protection & Water Quality Monitoring**

### Research Gap Addressed
All established ecological risk indices (PLI, PERI, PHI, RQ) require polymer chemical identification via FTIR spectroscopy — a $20,000+ lab instrument that cannot be deployed in the field. Our Contextual Threat Score (CTS) is the first morphology-and-geometry-derived risk framework computable directly from a microscope image, with scores dynamically adjusted by marine zone and seasonal ecology. This enables real-time field assessment without any chemical analysis.
 

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

### Preprocessing 
1. **Auto-Orient** — corrects EXIF rotation inconsistencies across microscope image exports
2. **Resize** — all images standardised to 640×640 pixels
3. **Adaptive Equalization (CLAHE)** — Contrast Limited Adaptive Histogram Equalization applied to normalise uneven microscope illumination (bright center, darker edges in circular field of view). Uses local contrast normalisation with clip limit 2.0 and tile grid 8×8, preserving inter-particle texture differences critical for morphology classification

### Augmentation
- Horizontal flip
- Vertical flip
- Rotation ±15°
- Brightness adjustment ±20%


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
## 3. Risk Scoring — Contextual Threat Score (CTS)
 
The CTS is a multi-layer scoring system that combines per-particle geometry, biological pathway analysis, depth zone physics, and contextual ecology (marine zone × season) into a single 0–100 threat score with a natural language narrative.
 
### Layer 0 — Zone × Season Context
 
Scores are not static. Weights and severity multipliers shift based on the marine zone being assessed and the active ecological season, reflecting real-world variation in species vulnerability.
 
**5 supported marine zones:**
 
| Zone | Supported seasons |
|---|---|
| `coral_reef` | spawning, dry, monsoon, winter, migration, upwelling |
| `mangrove_estuary` | spawning, dry, monsoon, winter, migration, upwelling |
| `pelagic_open_ocean` | spawning, dry, monsoon, winter, migration, upwelling, melt, ice |
| `coastal_benthic` | spawning, dry, monsoon, winter, migration, upwelling |
| `polar` | melt, ice, migration, spawning |
 
Each zone-season combination provides dynamic weights (w1, w2, w3) for sub-scores S, P, H — a severity loading factor L — and a species-at-risk list per depth zone.
 
*Example — coral reef during spawning season: w1=0.40, w2=0.35, w3=0.25, L=2.2. Species at risk: sea turtle, reef fish larvae, coral polyps, sea urchin, juvenile parrotfish.*
 
---
 
### Sub-score S — Toxin Load (SA:V proxy)
 
S estimates the surface-area-to-volume ratio of each particle from 2D geometry, which governs how efficiently a particle adsorbs and leaches persistent organic pollutants (POPs) and heavy metals.
 
**Morphology-specific SA:V formulas:**
 
| Morphology | Formula | Rationale |
|---|---|---|
| fiber | `4 / (min_feret_mm)` | Cylindrical approximation — thin fibers have extreme SA:V |
| pallet | `6 / (max_feret_mm)` | Sphere approximation — standard for pellets |
| fragment | `(perimeter / area) / convexity × rugosity` | Irregular surface — rugosity amplifies exposed area |
| film | `perimeter / area` | Thin sheet — perimeter-to-area captures flatness |
 
**Size modifier applied to all morphologies:**
 
| Particle size | Modifier | Rationale |
|---|---|---|
| < 100 µm | 1.4× | Nanoscale — highest cellular uptake probability |
| 100–1000 µm | 1.2× | Mesoscale — organ-level ingestion |
| 1000–5000 µm | 1.0× | Standard microplastic range |
| > 5000 µm | 0.7× | Macroplastic — lower bioavailability |
 
S is normalised to a 0–10 scale.
 
---
 
### Sub-score P — Biological Pathway
 
P identifies the specific biological harm mechanism based on particle shape geometry, scored 0–10. Geometry takes precedence over morphology class.
 
| Geometric condition | Pathway | Score |
|---|---|---|
| circularity > 0.85 AND size < 100µm | Cellular penetration | 10.0 |
| aspect ratio > 10 | Gill entanglement | 9.0 |
| circularity < 0.4 AND 100µm < size < 3000µm | Digestive lodging | 7.0 |
| size > 2000µm AND circularity < 0.3 | External entanglement | 6.0 |
| fallback by morphology class | Class default | 6–9 |
 
**Class defaults (fallback when geometry is ambiguous):**
 
| Class | Pathway | P |
|---|---|---|
| fiber | gill entanglement | 9 |
| fragment | digestive lodging | 7 |
| film | external entanglement | 6 |
| pallet | false satiation | 7 |
 
---
 
### Sub-score H — Depth Zone / Buoyancy
 
H determines where in the water column a particle settles, dictating which organism communities are exposed.
 
**Assignment logic (geometry-driven):**
- aspect ratio > 8 OR area < 500µm² → `surface_drift` (fibers and tiny particles float)
- area > 5,000,000µm² → `surface_drift` (very large films float)
- otherwise → morphology class default
 
| Depth zone | H score | Representative organisms |
|---|---|---|
| surface_drift | 8.5 | seabirds, sea turtles, surface fish |
| mid_column | 6.0 | juvenile fish, squid, filter feeders |
| benthic | 5.0 | coral, mussels, crabs, flatfish |
 
---
 
### Per-particle Score (Tp)
 
Each particle receives a raw score combining the three sub-scores with zone-season weights, then scaled by YOLO detection confidence:
 
```
Tp_raw = S × w1 + P × w2 + H × w3
Tp     = Tp_raw × confidence
```
 
Confidence weighting means low-certainty detections contribute proportionally less to the final sample score — linking model uncertainty directly to ecological risk.
 
---
 
### Sample-level Aggregation
 
The sample score weights the top-risk particles more heavily, reflecting that a few highly dangerous particles disproportionately drive ecological harm.
 
```
T_base = top_20%_mean × 0.7 + full_mean × 0.3
 
D (density modifier)  = min(2.5,  1 + log10(particles / cm²) × 0.6)
 
diversity_mod         = min(1.3,  1 + Shannon_entropy × 0.15)
  # Shannon entropy across morphology classes
  # mixed-morphology samples score higher — multiple harm pathways simultaneously active
 
T_raw   = T_base × D × diversity_mod × L
T_final = min(100, round(T_raw / 35 × 100))
```
 
**Severity bands:**
 
| Score | Band |
|---|---|
| 81–100 | Critical |
| 61–80 | High |
| 41–60 | Moderate |
| 21–40 | Low |
| 0–20 | Negligible |
 
---
 


## 4. Key Features
 
- **Image input** — accepts `.jpg` / `.png` microscope or high-magnification smartphone images
- **Morphology classification** — detects and classifies all particles per image into fiber, film, fragment, or pallet using YOLOv11s (multiple particles per image supported)
- **Size & geometry estimation** — computes Feret diameter, area, perimeter, circularity, aspect ratio, convexity, and rugosity per particle from contour analysis
- **Contextual Threat Score (CTS)** — zone × season aware risk score with S/P/H sub-scores, density modifier, Shannon diversity modifier, and natural language narrative 
---
 
## 5. Solution Architecture
 
```
Input Image
    ↓
Preprocessing (CLAHE → resize to 640×640)
    ↓
YOLOv11s Detection
    ↓
Bounding boxes + class + confidence
    ↓                         ↓
Size & Geometry           Grad-CAM Heatmap
Estimation                (per-particle overlay)
(Feret, area,
 circularity,
 aspect ratio,
 convexity, rugosity)
    ↓
Contextual Threat Score (CTS)
  ├── S: Toxin load (SA:V proxy from geometry)
  ├── P: Biological pathway (geometry-driven)
  ├── H: Depth zone / buoyancy
  ├── Zone × Season weights (Layer 0 — 5 zones, 8 seasons)
  ├── Density modifier D
  ├── Shannon diversity modifier
  └── Narrative generator
    ↓
Frontend Dashboard
(annotated image · per-particle table · CTS gauge ·
 species at risk · narrative · source attribution report)
```
 
---



### 5.Risk score metrics
<img width="1030" height="540" alt="Screenshot 2026-03-28 142604" src="https://github.com/user-attachments/assets/1bb2d070-e888-4c9b-8ab9-3514f839eede" />


---

### 6. Streamlit Web Interface
<img width="739" height="779" alt="Screenshot 2026-03-28 at 20 09 52" src="https://github.com/user-attachments/assets/5519d456-e5b6-474d-971b-cc13ce9c239f" />

Feret Metrics 

<img width="723" height="402" alt="Screenshot 2026-03-28 at 20 05 46" src="https://github.com/user-attachments/assets/488b195d-3e41-43fe-b174-9adcc777c3a6" />


## 7. References

1. Cambridge Prisms: Microplastics — Zooplankton ingestion study, 2023
2. ACS Environmental Science & Technology — Selective microplastic ingestion by copepod species, 2020
3. Frontiers in Marine Science — Biofouling and morphology effects on MP fate, 2022
4. PMC Aquatic Ecosystems — Marine food web trophic transfer of MPs, 2024
5. USGS Strategic Science Vision — Microplastic source attribution gaps, 2023

---

*Dataset: [Roboflow Universe — microplastic_100 by kueranan](https://universe.roboflow.com/kueranan/microplastic_100)*
*Model weights: YOLOv11s fine-tuned, available as `best.pt`*
