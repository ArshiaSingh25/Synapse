import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
from size_estimation import get_pixels_per_um, process_detections
from ultralytics import YOLO

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load YOLO model once and cache it for the session."""
    return YOLO("best.pt")   # ← place your trained best.pt in the same folder as app.py

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MicroScan — Microplastic Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #1e2640;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin: 0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.75rem;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #111827;
    border: 1px solid #1e2640;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.4rem;
}

/* Class badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-fiber    { background: #1a1a2e; color: #ff6b9d; border: 1px solid #ff6b9d; }
.badge-fragment { background: #1a1a2e; color: #ff9f43; border: 1px solid #ff9f43; }
.badge-film     { background: #1a1a2e; color: #48dbfb; border: 1px solid #48dbfb; }
.badge-pellet   { background: #1a1a2e; color: #a29bfe; border: 1px solid #a29bfe; }

/* Risk meter */
.risk-bar-container {
    background: #1e2640;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.risk-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.risk-critical { background: linear-gradient(90deg, #ff4757, #ff6b81); }
.risk-high     { background: linear-gradient(90deg, #ff9f43, #ffd32a); }
.risk-moderate { background: linear-gradient(90deg, #48dbfb, #00d4ff); }
.risk-low      { background: linear-gradient(90deg, #2ed573, #7bed9f); }

/* Severity label */
.severity-critical { color: #ff4757; font-weight: 700; }
.severity-high     { color: #ff9f43; font-weight: 700; }
.severity-moderate { color: #48dbfb; font-weight: 700; }
.severity-low      { color: #2ed573; font-weight: 700; }

/* Table */
.dataframe { background: #111827 !important; }

/* Upload zone */
.upload-zone {
    border: 2px dashed #1e2640;
    border-radius: 12px;
    padding: 3rem;
    text-align: center;
    background: #0d1117;
    margin-bottom: 1rem;
}
.upload-icon { font-size: 3rem; margin-bottom: 1rem; }
.upload-text { color: #6b7280; font-size: 0.9rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2640;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label {
    color: #9ca3af !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Breakdown row */
.breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e2640;
    font-size: 0.85rem;
}
.breakdown-label { color: #9ca3af; }
.breakdown-value { font-family: 'Space Mono', monospace; color: #e8eaf0; }

/* Particle result card */
.particle-card {
    background: #0d1117;
    border: 1px solid #1e2640;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.particle-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

/* Info box */
.info-box {
    background: #0d1420;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #9ca3af;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e2640;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

CLASS_MAP = {
    0: "Fiber",
    1: "Fragment",
    2: "Film",
    3: "Pellet",
}

CLASS_COLORS = {
    "Fiber":    (255, 107, 157),
    "Fragment": (255, 159, 67),
    "Film":     (72, 219, 251),
    "Pellet":   (162, 155, 254),
}

CLASS_INFO = {
    "Fiber":    "Thread-like. Ingested by filter-feeders, entangles digestive tract.",
    "Fragment": "Jagged, irregular. High surface area leaches chemical additives.",
    "Film":     "Thin, sheet-like. Smothers benthic organisms on the seafloor.",
    "Pellet":   "Spherical, uniform. Mistaken for fish eggs, bioaccumulates up the food chain.",
}

def get_severity(score):
    if score > 75: return "Critical", "critical"
    if score > 50: return "High",     "high"
    if score > 25: return "Moderate", "moderate"
    return "Low", "low"

def preprocess(crop):
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def true_max_feret(contour):
    hull   = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) < 2:
        x, y, w, h = cv2.boundingRect(contour)
        return float(max(w, h)), None, None
    max_d, p1, p2 = 0.0, hull[0], hull[1]
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = float(np.linalg.norm(hull[i] - hull[j]))
            if d > max_d:
                max_d, p1, p2 = d, hull[i], hull[j]
    return max_d, p1, p2

def extract_metrics(contour, bbox, pixels_per_um):
    x, y, w, h = bbox
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    feret_px, p1, p2 = true_max_feret(contour)

    if len(contour) >= 5:
        rect        = cv2.minAreaRect(contour)
        min_feret_px = min(rect[1]) if min(rect[1]) > 0 else 1
    else:
        min_feret_px = float(min(w, h))

    aspect_ratio = feret_px / min_feret_px if min_feret_px > 0 else 1
    circularity  = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0
    extent       = area / (w * h)  if (w * h) > 0  else 0
    equiv_diam   = np.sqrt(4 * area / np.pi)

    pum = pixels_per_um
    return {
        "feret_um":          round(feret_px   / pum, 2),
        "min_feret_um":      round(min_feret_px / pum, 2),
        "aspect_ratio":      round(aspect_ratio, 2),
        "circularity":       round(circularity,  3),
        "solidity":          round(solidity,      3),
        "extent":            round(extent,        3),
        "equiv_diameter_um": round(equiv_diam   / pum, 2),
        "area_um2":          round(area         / pum**2, 2),
        "feret_p1":          p1,
        "feret_p2":          p2,
    }

def infer_entry_route(feret_um, circularity):
    if feret_um < 1:    return "Cell membrane penetration", 100
    if feret_um < 10:   return "Tissue absorption",         90
    if feret_um < 100:
        return ("Gut + gill absorption", 75) if circularity < 0.3 else ("Ingestion", 65)
    if feret_um < 1000: return "Ingestion",                 50
    return "Surface contact only", 20

def infer_destination(feret_um):
    if feret_um < 1:    return "Bloodstream",      100
    if feret_um < 10:   return "Organ tissue",      88
    if feret_um < 100:  return "Gut / Gill",        65
    if feret_um < 500:  return "Digestive tract",   40
    return "Surface / expelled", 15

def sav_score(cls, feret_um, min_feret_um):
    t = min_feret_um if min_feret_um > 0 else max(feret_um * 0.1, 0.01)
    if   cls == "Fiber":    raw = 2   / (t / 2)
    elif cls == "Film":     raw = 2   / t
    elif cls == "Fragment": raw = 6   / t
    else:                   raw = 6   / feret_um
    return round(min(100, raw * 10), 1)

def structural_complexity(solidity, circularity):
    return round((1 - solidity) * 60 + (1 - circularity) * 40, 1)

def density_proxy(cls, extent, solidity):
    base = {"Pellet": 90, "Fragment": 70, "Fiber": 55, "Film": 40}.get(cls, 60)
    return round(base * (extent + solidity) / 2, 1)

def calculate_risk(cls, metrics, eco_mult=1.0, season_mult=1.0):
    feret    = metrics["feret_um"]
    min_f    = metrics["min_feret_um"]
    circ     = metrics["circularity"]
    sol      = metrics["solidity"]
    ext      = metrics["extent"]

    class_base   = {"Fiber": 80, "Fragment": 65, "Film": 55, "Pellet": 50}.get(cls, 55)
    sav          = sav_score(cls, feret, min_f)
    entry_label, entry_score = infer_entry_route(feret, circ)
    dest_label,  dest_score  = infer_destination(feret)
    complexity   = structural_complexity(sol, circ)
    density      = density_proxy(cls, ext, sol)

    base = (
        class_base  * 0.20 +
        sav         * 0.25 +
        entry_score * 0.20 +
        dest_score  * 0.20 +
        complexity  * 0.10 +
        density     * 0.05
    )
    final = round(min(100, base * eco_mult * season_mult), 1)
    severity, sev_key = get_severity(final)

    return {
        "final_score": final,
        "severity":    severity,
        "sev_key":     sev_key,
        "breakdown": {
            "Class base":            class_base,
            "Toxin load (SA:V)":     sav,
            "Entry route":           f"{entry_label} ({entry_score})",
            "Destination":           f"{dest_label} ({dest_score})",
            "Structural complexity": complexity,
            "Density proxy":         density,
            "Ecosystem multiplier":  f"x{eco_mult:.2f}",
            "Season multiplier":     f"x{season_mult:.2f}",
        }
    }

def analyze_image(img_bgr, pixels_per_um, eco_mult=1.0, season_mult=1.0, mock_classes=None):
    """
    Real contour-based analysis.
    mock_classes: list of class names to assign per contour (simulates model output)
    """
    thresh   = preprocess(img_bgr)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 300]

    annotated = img_bgr.copy()
    results   = []

    for i, cnt in enumerate(contours):
        # Assign class — real app uses model output; here we cycle through classes
        if mock_classes and i < len(mock_classes):
            cls = mock_classes[i]
        else:
            cls_list = ["Fiber", "Fragment", "Film", "Pellet"]
            cls = cls_list[i % len(cls_list)]

        bbox    = cv2.boundingRect(cnt)
        x, y, w, h = bbox
        metrics = extract_metrics(cnt, bbox, pixels_per_um)
        risk    = calculate_risk(cls, metrics, eco_mult, season_mult)

        color_rgb = CLASS_COLORS[cls]
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        # Draw contour
        cv2.drawContours(annotated, [cnt], -1, color_bgr, 2)

        # Draw Feret diameter line
        if metrics["feret_p1"] is not None and metrics["feret_p2"] is not None:
            p1 = (int(metrics["feret_p1"][0]) + x, int(metrics["feret_p1"][1]) + y)
            p2 = (int(metrics["feret_p2"][0]) + x, int(metrics["feret_p2"][1]) + y)
            cv2.line(annotated, p1, p2, color_bgr, 1)

        # Label
        label = f"{cls[0]} | {risk['final_score']}"
        cv2.rectangle(annotated, (x, y - 18), (x + len(label)*8, y), color_bgr, -1)
        cv2.putText(annotated, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 20), 1)

        results.append({
            "id":    i + 1,
            "class": cls,
            **metrics,
            **risk,
        })

    return annotated, results



# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace;color:#00d4ff;
                font-size:1rem;letter-spacing:1px;padding:1rem 0 0.5rem 0;'>
        🔬 MICROSCAN
    </div>
    <div style='font-size:0.7rem;color:#6b7280;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:1.5rem;'>
        Microplastic Analyzer v1.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📏 Calibration")
    user_type = st.radio(
        "Image source",
        ["Lab image — manual entry",
         "Smartphone with clip-on lens"],
        label_visibility="collapsed"
    )

    pixels_per_um = 1.0
    if user_type == "Lab image — manual entry":
        st.markdown('<div class="info-box">Enter the pixels per µm for your microscope setup. Check your microscope manual or measure a known reference.</div>',
                    unsafe_allow_html=True)
        pum = st.number_input("Pixels per µm", min_value=0.001, value=1.0, step=0.1)
        pixels_per_um = get_pixels_per_um("manual", pixels_per_um=pum)

    else:  # Smartphone with clip-on lens
        lens = st.selectbox("Clip-on lens magnification",
                            ["No attachment", "10x", "20x", "60x"])
        pixels_per_um = get_pixels_per_um("smartphone", lens=lens)
        st.markdown('<div class="info-box">Approximate values. Actual µm size may vary by device.</div>',
                    unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#0d1420;border-radius:8px;padding:0.75rem;
                margin-top:0.5rem;font-family:Space Mono,monospace;
                font-size:0.8rem;color:#00d4ff;'>
        1 px = {1/pixels_per_um:.3f} µm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🌍 Sample Context")

    ecosystem = st.selectbox("Ecosystem type", [
        "Drinking water supply",
        "Coral reef",
        "Deep sea sediment",
        "River / Estuary",
        "Open ocean",
        "Coastal / Beach",
        "Lake / Freshwater",
        "Mangrove / Wetland",
    ])

    season = st.selectbox("Season of collection", [
        "Post-monsoon",
        "Monsoon / Wet season",
        "Summer / Dry season",
        "Winter",
        "Spring",
        "Autumn",
    ])

    # Multipliers for risk score
    ECOSYSTEM_MULTIPLIER = {
        "Drinking water supply": 1.5,
        "Coral reef":            1.35,
        "Deep sea sediment":     1.25,
        "Mangrove / Wetland":    1.20,
        "River / Estuary":       1.10,
        "Open ocean":            1.00,
        "Lake / Freshwater":     0.95,
        "Coastal / Beach":       0.90,
    }

    # Seasons affect biological activity & organism exposure
    SEASON_MULTIPLIER = {
        "Monsoon / Wet season":  1.20,  # runoff carries more microplastics
        "Post-monsoon":          1.15,  # peak accumulation after runoff
        "Summer / Dry season":   1.05,  # fragmentation accelerates in UV
        "Spring":                1.00,
        "Autumn":                0.95,
        "Winter":                0.90,  # lower biological activity
    }

    eco_mult    = ECOSYSTEM_MULTIPLIER.get(ecosystem, 1.0)
    season_mult = SEASON_MULTIPLIER.get(season, 1.0)

    eco_info = {
        "Drinking water supply": "Direct human exposure pathway.",
        "Coral reef":            "Fragile ecosystem, high biodiversity.",
        "Deep sea sediment":     "Particles accumulate — no escape route.",
        "Mangrove / Wetland":    "Nursery ground for many species.",
        "River / Estuary":       "High biodiversity, active transport zone.",
        "Open ocean":            "Baseline risk environment.",
        "Lake / Freshwater":     "Enclosed system, slower dilution.",
        "Coastal / Beach":       "Less biologically active water column.",
    }
    season_info = {
        "Monsoon / Wet season":  "Peak runoff — highest microplastic input.",
        "Post-monsoon":          "Accumulated load post-runoff.",
        "Summer / Dry season":   "UV fragmentation accelerates.",
        "Spring":                "Moderate biological activity.",
        "Autumn":                "Declining biological activity.",
        "Winter":                "Lowest organism exposure.",
    }

    st.markdown(f'<div class="info-box">📍 {eco_info[ecosystem]}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">🗓️ {season_info[season]}</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:#0d1420;border-radius:8px;padding:0.75rem;
                margin-top:0.5rem;font-family:Space Mono,monospace;
                font-size:0.75rem;color:#ff9f43;'>
        Risk multiplier: ×{eco_mult:.2f} (eco) · ×{season_mult:.2f} (season)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Display")
    show_breakdown = st.toggle("Show risk breakdown",    value=True)
    show_table     = st.toggle("Show metrics table",     value=True)
    batch_mode     = st.toggle("Batch processing mode",  value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#374151;line-height:1.6;'>
        Color legend:<br>
        <span style='color:#ff6b9d;'>■</span> Fiber &nbsp;
        <span style='color:#ff9f43;'>■</span> Fragment<br>
        <span style='color:#48dbfb;'>■</span> Film &nbsp;&nbsp;
        <span style='color:#a29bfe;'>■</span> Pellet
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🔬 MicroScan</div>
        <div class="app-subtitle">Microplastic Morphology Classifier · Marine Ecosystem Protection</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload microscope image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=batch_mode,
    label_visibility="collapsed"
)

if not uploaded_files:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🧫</div>
        <div style='font-family:Space Mono,monospace;color:#e8eaf0;
                    font-size:1rem;margin-bottom:0.5rem;'>
            Drop microscope images here
        </div>
        <div class="upload-text">
            Accepts .jpg / .png · Lab images or high-magnification smartphone photos<br>
            Classifies into Fiber · Fragment · Film · Pellet
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Class reference cards
    st.markdown("### Morphological Classes")
    cols = st.columns(4)
    icons = {"Fiber": "🧵", "Fragment": "🪨", "Film": "🎞️", "Pellet": "🔵"}
    for col, (cls, info) in zip(cols, CLASS_INFO.items()):
        color = f"rgb{CLASS_COLORS[cls]}"
        with col:
            st.markdown(f"""
            <div class="card" style="border-color:{color}22;">
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>{icons[cls]}</div>
                <div style='font-family:Space Mono,monospace;color:{color};
                            font-size:0.85rem;font-weight:700;margin-bottom:0.5rem;'>
                    {cls.upper()}
                </div>
                <div style='font-size:0.8rem;color:#9ca3af;line-height:1.5;'>
                    {info}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ── Process each file ──────────────────────────────────────────────────────────
if not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]

all_results = []

for uploaded_file in uploaded_files:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error(f"Could not read {uploaded_file.name}")
        continue

    with st.spinner(f"Running model on {uploaded_file.name}…"):
        model      = load_model()
        yolo_results = model(img_bgr)[0]   # inference on BGR numpy array

        # Convert YOLO results → list of {"class", "bbox": (x,y,w,h), "confidence"}
        detections = []
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class":      CLASS_MAP.get(cls_id, "Unknown"),
                "bbox":       (x1, y1, x2 - x1, y2 - y1),   # x,y,w,h
                "confidence": round(conf, 3),
            })

    with st.spinner(f"Estimating sizes…"):
        annotated, results = process_detections(img_bgr, detections, pixels_per_um)
        # Add risk scores
        for r in results:
            risk = calculate_risk(r["class"], {
                "feret_um":     r["max_feret_um"],
                "min_feret_um": r["min_feret_um"],
                "circularity":  r["circularity"],
                "solidity":     r["solidity"],
                "extent":       r["extent"],
            }, eco_mult, season_mult)
            r.update(risk)
            r["feret_um"] = r["max_feret_um"]   # UI compatibility

    if len(uploaded_files) > 1:
        st.markdown(f"### 📄 {uploaded_file.name}")

    if not results:
        st.warning("No particles detected. Try adjusting image contrast or calibration.")
        continue

    # ── Image columns ──────────────────────────────────────────────────────────
    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.markdown('<div class="card-title">Original Image</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_ann:
        st.markdown('<div class="card-title">Detected Particles</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Summary metrics ────────────────────────────────────────────────────────
    avg_risk    = np.mean([r["final_score"] for r in results])
    max_risk    = max(r["final_score"] for r in results)
    dominant    = pd.Series([r["class"] for r in results]).value_counts().index[0]
    avg_feret   = np.mean([r["feret_um"] for r in results])
    sev_label, sev_key = get_severity(avg_risk)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{len(results)}</div>
            <div class="metric-label">Particles Found</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color:#ff9f43;">{avg_risk:.1f}</div>
            <div class="metric-label">Avg Threat Index</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">{dominant}</div>
            <div class="metric-label">Dominant Class</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">{avg_feret:.1f}</div>
            <div class="metric-label">Avg Size (µm)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overall risk bar
    bar_class = f"risk-{sev_key}"
    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
        <div style='display:flex;justify-content:space-between;
                    font-size:0.8rem;margin-bottom:0.4rem;'>
            <span style='color:#9ca3af;font-family:Space Mono,monospace;
                         text-transform:uppercase;letter-spacing:1px;'>
                Ecological Threat Index
            </span>
            <span class='severity-{sev_key}'>{sev_label} · {avg_risk:.1f}/100</span>
        </div>
        <div class="risk-bar-container">
            <div class="risk-bar {bar_class}" style="width:{avg_risk}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-particle results ───────────────────────────────────────────────────
    st.markdown('<div class="card-title">Per-Particle Analysis</div>',
                unsafe_allow_html=True)

    for r in results:
        cls       = r["class"]
        score     = r["final_score"]
        sev, skey = get_severity(score)
        color     = f"rgb{CLASS_COLORS[cls]}"

        with st.expander(
            f"Particle #{r['id']} · {cls} · ETI {score} · {sev}",
            expanded=(len(results) <= 3)
        ):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown(f"""
                <div style='margin-bottom:0.75rem;'>
                    <span class='badge badge-{cls.lower()}'>{cls.upper()}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Max Feret Diameter</span>
                    <span class="breakdown-value">{r['feret_um']} µm</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Min Feret Diameter</span>
                    <span class="breakdown-value">{r['min_feret_um']} µm</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Aspect Ratio</span>
                    <span class="breakdown-value">{r['aspect_ratio']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Circularity</span>
                    <span class="breakdown-value">{r['circularity']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Solidity</span>
                    <span class="breakdown-value">{r['solidity']}</span>
                </div>
                <div class="breakdown-row">
                    <span class="breakdown-label">Area</span>
                    <span class="breakdown-value">{r['area_um2']} µm²</span>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                if show_breakdown:
                    st.markdown(f"""
                    <div style='font-family:Space Mono,monospace;font-size:0.7rem;
                                color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:0.75rem;'>
                        Risk Breakdown
                    </div>
                    """, unsafe_allow_html=True)
                    for param, val in r["breakdown"].items():
                        st.markdown(f"""
                        <div class="breakdown-row">
                            <span class="breakdown-label">{param}</span>
                            <span class="breakdown-value">{val}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    bar_w  = min(100, score)
                    st.markdown(f"""
                    <div style='margin-top:1rem;'>
                        <div style='display:flex;justify-content:space-between;
                                    font-size:0.75rem;margin-bottom:0.4rem;'>
                            <span style='color:#9ca3af;'>Ecological Threat Index</span>
                            <span class='severity-{skey}'>{score} / 100</span>
                        </div>
                        <div class="risk-bar-container">
                            <div class="risk-bar risk-{skey}"
                                 style="width:{bar_w}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Metrics table ──────────────────────────────────────────────────────────
    if show_table:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">Full Metrics Table</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame([{
            "ID":           r["id"],
            "Class":        r["class"],
            "Feret (µm)":   r["feret_um"],
            "Min Feret":    r["min_feret_um"],
            "Aspect Ratio": r["aspect_ratio"],
            "Circularity":  r["circularity"],
            "Solidity":     r["solidity"],
            "Area (µm²)":   r["area_um2"],
            "ETI Score":    r["final_score"],
            "Severity":     r["severity"],
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

    all_results.extend(results)

    if len(uploaded_files) > 1:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Batch summary ──────────────────────────────────────────────────────────────
if batch_mode and len(all_results) > 0 and len(uploaded_files) > 1:
    st.markdown("## 📊 Batch Summary Report")
    df_all = pd.DataFrame(all_results)

    b1, b2, b3 = st.columns(3)
    b1.metric("Total Particles",   len(df_all))
    b2.metric("Overall Avg ETI",   f"{df_all['final_score'].mean():.1f}")
    b3.metric("Highest Risk",      f"{df_all['final_score'].max():.1f}")

    st.markdown("**Class Distribution**")
    dist = df_all["class"].value_counts().reset_index()
    dist.columns = ["Class", "Count"]
    st.bar_chart(dist.set_index("Class"))