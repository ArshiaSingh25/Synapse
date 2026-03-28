import cv2
import numpy as np

CLASS_MAP = {
    0: "Fiber",
    1: "Fragment",
    2: "Film",
    3: "Pellet"
}

COLOR_MAP = {
    "Fiber":    (157, 107, 255),
    "Fragment": ( 67, 159, 255),
    "Film":     (251, 219,  72),
    "Pellet":   (254, 155, 162),
}

# ── Calibration ────────────────────────────────────────────────────────────────

def get_pixels_per_um(mode, **kwargs):
    """
    Called by app.py to compute pixels_per_um from user input.
    The result is then passed into process_detections().

    mode = "scale_bar"  → user measured scale bar in image
    mode = "manual"     → user directly entered pixels per µm
    mode = "smartphone" → user selected clip-on lens type

    Examples:
        get_pixels_per_um("scale_bar", scale_px=200, scale_um=500)  → 0.4
        get_pixels_per_um("manual", pixels_per_um=2.0)              → 2.0
        get_pixels_per_um("smartphone", lens="20x")                 → 1.8
    """
    if mode == "scale_bar":
        scale_px = kwargs.get("scale_px", 100)
        scale_um = kwargs.get("scale_um", 500)
        if scale_um <= 0:
            raise ValueError("scale_um must be > 0")
        return scale_px / scale_um

    elif mode == "manual":
        pum = kwargs.get("pixels_per_um", 1.0)
        if pum <= 0:
            raise ValueError("pixels_per_um must be > 0")
        return pum

    elif mode == "smartphone":
        lens_map = {
            "No attachment": 0.2,
            "10x":           0.8,
            "20x":           1.8,
            "60x":           5.0,
        }
        lens = kwargs.get("lens", "No attachment")
        if lens not in lens_map:
            raise ValueError(f"Unknown lens: {lens}. Choose from {list(lens_map.keys())}")
        return lens_map[lens]

    else:
        raise ValueError(f"Unknown calibration mode: '{mode}'. Use 'scale_bar', 'manual', or 'smartphone'.")


# ── Step 1: Validate detections from YOLO model ───────────────────────────────

def validate_detections(detections, img_w, img_h):
    """
    Accepts detections already parsed by app.py from YOLO output.
    Each detection: {"class": str, "bbox": (x, y, w, h), "confidence": float}

    Clamps bboxes to image bounds and filters out invalid ones.
    Returns cleaned list of detections.
    """
    valid = []
    for det in detections:
        cls        = det.get("class", "Unknown")
        x, y, w, h = det["bbox"]

        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            continue

        valid.append({
            "class":      cls,
            "bbox":       (x, y, w, h),
            "confidence": det.get("confidence", 1.0),
        })

    return valid


# ── Step 2: Preprocess with CLAHE ──────────────────────────────────────────────

def preprocess_clahe(crop):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) corrects
    uneven microscope illumination locally before thresholding.
    Produces cleaner, more accurate contours than plain Otsu.
    """
    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ── Step 3: Watershed ──────────────────────────────────────────────────────────

def apply_watershed(crop_bgr, thresh):
    """
    Separates touching/overlapping particles that simple thresholding
    would merge into one blob. Falls back to plain contours if watershed
    finds nothing (e.g. isolated, non-touching particles).
    """
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    if dist.max() == 0:
        # Blank region — fallback immediately
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        return list(cnts)

    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    markers    = cv2.watershed(crop_bgr.copy(), markers)

    contours = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours.append(max(cnts, key=cv2.contourArea))

    if not contours:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = list(cnts)

    return contours


# ── Step 4: True Max Feret ─────────────────────────────────────────────────────

def true_max_feret(contour):
    """
    Rotating calipers on the convex hull.
    Returns the true maximum Feret diameter — more accurate than
    minAreaRect for irregular/concave shapes like fragments.
    Returns (max_dist_px, point1, point2)
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)

    if len(hull) < 2:
        x, y, w, h = cv2.boundingRect(contour)
        return float(max(w, h)), None, None

    max_dist = 0.0
    p1_best, p2_best = hull[0], hull[1]

    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = float(np.linalg.norm(hull[i] - hull[j]))
            if d > max_dist:
                max_dist          = d
                p1_best, p2_best  = hull[i], hull[j]

    return max_dist, p1_best, p2_best


# ── Step 5: Full Morphometric Profile ─────────────────────────────────────────

def full_morphometric_profile(contour, pixels_per_um, offset=(0, 0)):
    """
    Extracts all geometric metrics from a contour.

    pixels_per_um : conversion factor from app.py calibration
    offset        : (x, y) top-left of crop in full image coords,
                    so Feret line points are placed correctly on the
                    annotated image.

    Returns a dict of size metrics (µm) + shape descriptors.
    Returns None if the contour is too small (noise).
    """
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area < 10:
        return None

    # True max Feret diameter
    max_feret_px, p1, p2 = true_max_feret(contour)

    # Convert Feret endpoints to full-image coordinates
    if p1 is not None:
        p1_global = (int(p1[0]) + offset[0], int(p1[1]) + offset[1])
        p2_global = (int(p2[0]) + offset[0], int(p2[1]) + offset[1])
    else:
        p1_global = p2_global = None

    # Min Feret via minAreaRect (tight rotated rectangle)
    if len(contour) >= 5:
        rect         = cv2.minAreaRect(contour)
        min_feret_px = min(rect[1]) if min(rect[1]) > 0 else 1.0
        angle        = rect[2]
    else:
        _, _, bw, bh = cv2.boundingRect(contour)
        min_feret_px = float(min(bw, bh))
        angle        = 0.0

    aspect_ratio = max_feret_px / min_feret_px if min_feret_px > 0 else 1.0
    circularity  = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0.0

    _, _, bw, bh = cv2.boundingRect(contour)
    extent       = area / (bw * bh) if (bw * bh) > 0 else 0.0

    equiv_diam_px = np.sqrt(4 * area / np.pi)

    pum = pixels_per_um

    return {
        # ── Size metrics in µm ──────────────────────────────────────────────
        "max_feret_um":      round(max_feret_px  / pum, 2),  # KEY metric
        "min_feret_um":      round(min_feret_px  / pum, 2),
        "feret_ratio":       round(aspect_ratio,         2),  # max/min
        "equiv_diameter_um": round(equiv_diam_px / pum, 2),
        "area_um2":          round(area          / pum**2, 2),
        "perimeter_um":      round(perimeter     / pum, 2),
        # ── Shape descriptors (dimensionless) ───────────────────────────────
        "aspect_ratio":      round(aspect_ratio,  2),
        "circularity":       round(circularity,   3),   # 1.0 = perfect circle
        "solidity":          round(solidity,      3),   # 1.0 = convex, low = jagged
        "extent":            round(extent,        3),   # fill ratio of bounding box
        "orientation_deg":   round(angle,         1),
        # ── For drawing on image ─────────────────────────────────────────────
        "feret_p1":          p1_global,
        "feret_p2":          p2_global,
    }


# ── Step 6: Main pipeline ──────────────────────────────────────────────────────

def process_detections(img_bgr, detections, pixels_per_um):
    """
    Full size estimation pipeline.

    Parameters
    ----------
    img_bgr       : np.ndarray   — BGR image from cv2
    detections    : list[dict]   — from app.py YOLO inference
                                   each: {"class", "bbox": (x,y,w,h), "confidence"}
    pixels_per_um : float        — from get_pixels_per_um() in app.py

    Returns
    -------
    annotated     : np.ndarray  — image with bounding boxes + Feret lines
    results       : list[dict]  — one dict per particle with all metrics
    """
    img_h, img_w = img_bgr.shape[:2]
    annotated    = img_bgr.copy()

    detections = validate_detections(detections, img_w, img_h)
    results    = []

    for i, det in enumerate(detections):
        cls         = det["class"]
        x, y, w, h  = det["bbox"]

        # Crop with padding
        pad = 5
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img_w, x + w + pad)
        y2  = min(img_h, y + h + pad)
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        thresh   = preprocess_clahe(crop)
        contours = apply_watershed(crop, thresh)

        if not contours:
            continue

        main_cnt = max(contours, key=cv2.contourArea)
        metrics  = full_morphometric_profile(
            main_cnt, pixels_per_um, offset=(x1, y1)
        )

        if metrics is None:
            continue

        # ── Annotate image ─────────────────────────────────────────────────
        color = COLOR_MAP.get(cls, (200, 200, 200))

        # Bounding box from YOLO
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # True Feret diameter line
        if metrics["feret_p1"] and metrics["feret_p2"]:
            cv2.line(annotated, metrics["feret_p1"], metrics["feret_p2"], color, 1)
            cv2.circle(annotated, metrics["feret_p1"], 3, color, -1)
            cv2.circle(annotated, metrics["feret_p2"], 3, color, -1)

        # Label: class initial + Feret size
        label = f"{cls[0]}  {metrics['max_feret_um']}um"
        cv2.rectangle(annotated, (x, y - 18),
                      (x + len(label) * 8, y), color, -1)
        cv2.putText(annotated, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 20), 1)

        results.append({
            "id":    i + 1,
            "class": cls,
            **metrics
        })

    return annotated, results