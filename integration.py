"""
integration.py
──────────────
Bridges your teammate's process_detections() output → compute_threat_score()

Their output per particle:
    {id, class, max_feret_um, min_feret_um, area_um2, perimeter_um,
     circularity, solidity, aspect_ratio, extent, orientation_deg,
     feret_p1, feret_p2, feret_ratio, equiv_diameter_um}

Your scorer expects per particle:
    {morph, confidence, geometry: {max_feret_um, min_feret_um, area_um2,
     perimeter_um, circularity, aspect_ratio, convexity, rugosity}}
"""

import cv2
import json
from risk_score import compute_threat_score


# ── Adapter: one particle dict ────────────────────────────────────────────────

def adapt_particle(det: dict, confidence: float = 1.0) -> dict:
    """
    Convert a single entry from process_detections() results list
    into the format expected by compute_threat_score().

    confidence: pass Roboflow's confidence value if available,
                defaults to 1.0 (YOLO txt has no confidence field)
    """
    morph = det["class"].lower()

    geometry = {
        "max_feret_um":  det["max_feret_um"],
        "min_feret_um":  det["min_feret_um"],
        "area_um2":      det["area_um2"],
        "perimeter_um":  det["perimeter_um"],
        "circularity":   det["circularity"],
        "aspect_ratio":  det["aspect_ratio"],
        "convexity":     det["solidity"],
        "rugosity":      round(1.0 + (1.0 - det["solidity"]) * 0.5, 3),
    }

    return {
        "morph":      morph,
        "confidence": confidence,
        "geometry":   geometry,
    }


# ── Main integration function ─────────────────────────────────────────────────

def run_full_pipeline(
    image_path:      str,
    txt_path:        str,
    zone:            str,
    season:          str,
    pixels_per_um:   float = 1.0,
    image_area_cm2:  float = 4.0,
    confidences:     dict  = None,
) -> dict:
    """
    End-to-end pipeline.

    Parameters
    ----------
    image_path     : path to microscope image
    txt_path       : YOLO .txt annotation file
    zone           : marine zone key  (e.g. "mangrove_estuary")
    season         : season key       (e.g. "monsoon")
    pixels_per_um  : calibration value from microscope
    image_area_cm2 : physical area the image covers (for density calc)
    confidences    : optional dict of {particle_id: float} from Roboflow API
                     if None, all particles default to confidence=1.0

    Returns
    -------
    Full risk score result dict from compute_threat_score(), plus
    the annotated image and raw CV results.
    """

    # ── Step 1: Load image ────────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": f"Could not load image at {image_path}"}

    img_h, img_w = img_bgr.shape[:2]

    # ── Step 1b: Parse YOLO txt into detections list ──────────────────────────
    from size_estimation import CLASS_MAP, process_detections

    detections = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx, cy, nw, nh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) > 5 else 1.0

            x = int((cx - nw / 2) * img_w)
            y = int((cy - nh / 2) * img_h)
            w = int(nw * img_w)
            h = int(nh * img_h)

            class_name = CLASS_MAP.get(class_id, "Fragment")

            detections.append({
                "class":      class_name,
                "bbox":       (x, y, w, h),
                "confidence": conf,
            })

    # ── Step 1c: Run teammate's CV pipeline ───────────────────────────────────
    annotated_img, cv_results = process_detections(
        img_bgr       = img_bgr,
        detections    = detections,
        pixels_per_um = pixels_per_um,
    )

    if not cv_results:
        return {
            "error":         "No particles detected in image",
            "annotated_img": annotated_img,
            "cv_results":    [],
        }

    # ── Step 2: Adapt CV output → scorer input ────────────────────────────────
    particles = []
    for det in cv_results:
        det_id = det["id"]
        conf = detections[det_id - 1]["confidence"] if det_id <= len(detections) else 1.0
        if confidences and det_id in confidences:
            conf = confidences[det_id]
        particles.append(adapt_particle(det, confidence=conf))

    # ── Step 3: Compute Ecological Threat Index ───────────────────────────────
    risk_result = compute_threat_score(
        particles      = particles,
        zone           = zone,
        season         = season,
        image_area_cm2 = image_area_cm2,
    )

    # ── Step 4: Attach CV metadata for downstream use ─────────────────────────
    risk_result["annotated_img"] = annotated_img
    risk_result["cv_results"]    = cv_results
    risk_result["input"] = {
        "image_path":    image_path,
        "zone":          zone,
        "season":        season,
        "pixels_per_um": pixels_per_um,
        "n_detected":    len(cv_results),
    }

    return risk_result


# ── Quick CLI test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    result = run_full_pipeline(
        image_path     = "sample.jpg",
        txt_path       = "sample.txt",
        zone           = "mangrove_estuary",
        season         = "monsoon",
        pixels_per_um  = 0.4,
        image_area_cm2 = 4.0,
    )

    if "error" in result:
        print(f"Pipeline error: {result['error']}")
    else:
        cv2.imwrite("annotated_output.jpg", result["annotated_img"])

        print(f"\n{'='*50}")
        print(f"  Ecological Threat Index : {result['T_final']} / 100")
        print(f"  Severity Band           : {result['band']}")
        print(f"  Particles scored        : {result['input']['n_detected']}")
        print(f"  Zone / Season           : {result['input']['zone']} / {result['input']['season']}")
        print(f"\n  Narrative:\n  {result['narrative']}")
        print(f"\n  Sub-scores: {result['sub_scores']}")
        print(f"  Species at risk: {', '.join(result['species_at_risk'])}")
        print(f"{'='*50}\n")

        exportable = {k: v for k, v in result.items()
                      if k not in ("annotated_img",)}
        with open("risk_result.json", "w") as f:
            json.dump(exportable, f, indent=2)
        print("Full result saved to risk_result.json")
