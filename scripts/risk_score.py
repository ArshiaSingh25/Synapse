#risk score
import math
from typing import Literal

# ── Layer 0: Zone × Season weight profiles ────────────────────────────────────
LAYER0 = {
    "coral_reef": {
        "spawning": {"w1":.40,"w2":.35,"w3":.25,"L":2.2,
            "species":{"surface_drift":["sea turtle","reef fish larvae"],"mid_column":["juvenile parrotfish","damselfish"],"benthic":["coral polyps","sea urchin"]},
            "note":"spawning season — larvae present, detox capacity near zero"},
        "dry":      {"w1":.32,"w2":.38,"w3":.30,"L":1.7,
            "species":{"surface_drift":["reef fish","sea turtle"],"mid_column":["parrotfish","angelfish"],"benthic":["coral polyps","starfish"]},
            "note":"dry season — stable reef, moderate vulnerability"},
        "monsoon":  {"w1":.30,"w2":.42,"w3":.28,"L":1.9,
            "species":{"surface_drift":["sea turtle","triggerfish"],"mid_column":["wrasse","surgeonfish"],"benthic":["sea urchin","clam"]},
            "note":"monsoon — particle accumulation on reef face elevated"},
        "winter":   {"w1":.30,"w2":.40,"w3":.30,"L":1.4,
            "species":{"surface_drift":["reef fish"],"mid_column":["damselfish"],"benthic":["coral polyps"]},
            "note":"winter — reduced metabolic activity"},
        "migration":{"w1":.33,"w2":.37,"w3":.30,"L":1.8,
            "species":{"surface_drift":["sea turtle","manta ray"],"mid_column":["reef shark","trevally"],"benthic":["coral polyps"]},
            "note":"migration corridor active"},
        "upwelling":{"w1":.38,"w2":.35,"w3":.27,"L":2.0,
            "species":{"surface_drift":["sea turtle","tuna"],"mid_column":["reef fish","squid"],"benthic":["sea urchin","coral polyps"]},
            "note":"upwelling brings benthic particles to surface"},
    },
    "mangrove_estuary": {
        "spawning": {"w1":.35,"w2":.43,"w3":.22,"L":2.1,
            "species":{"surface_drift":["mudskipper","heron"],"mid_column":["juvenile seabass","juvenile snapper"],"benthic":["fiddler crab","oyster"]},
            "note":"nursery season — most vulnerable juvenile life stage"},
        "dry":      {"w1":.35,"w2":.40,"w3":.25,"L":2.0,
            "species":{"surface_drift":["mudskipper","egret"],"mid_column":["juvenile snapper","mullet"],"benthic":["fiddler crab","oyster"]},
            "note":"dry season — dense juvenile fish nursery"},
        "monsoon":  {"w1":.35,"w2":.45,"w3":.20,"L":2.2,
            "species":{"surface_drift":["juvenile snapper","mudskipper"],"mid_column":["juvenile seabass"],"benthic":["fiddler crab","oyster"]},
            "note":"monsoon — particles accumulating in root shelter zones"},
        "winter":   {"w1":.33,"w2":.38,"w3":.29,"L":1.5,
            "species":{"surface_drift":["egret","mudskipper"],"mid_column":["mullet"],"benthic":["oyster","crab"]},
            "note":"winter — fewer juveniles, lower accumulation"},
        "migration":{"w1":.32,"w2":.42,"w3":.26,"L":1.8,
            "species":{"surface_drift":["shorebird","heron"],"mid_column":["juvenile snapper"],"benthic":["fiddler crab"]},
            "note":"shorebird migration — coastal feeding intensified"},
        "upwelling":{"w1":.38,"w2":.38,"w3":.24,"L":1.9,
            "species":{"surface_drift":["juvenile snapper","heron"],"mid_column":["seabass","mullet"],"benthic":["fiddler crab","oyster"]},
            "note":"upwelling-driven nutrient pulse"},
    },
    "pelagic_open_ocean": {
        "spawning": {"w1":.33,"w2":.32,"w3":.35,"L":1.7,
            "species":{"surface_drift":["seabird","sea turtle","flying fish"],"mid_column":["tuna","mahi-mahi"],"benthic":["deep sea fish"]},
            "note":"pelagic spawning — depth zone exposure critical"},
        "dry":      {"w1":.30,"w2":.30,"w3":.40,"L":1.5,
            "species":{"surface_drift":["seabird","sea turtle","tuna"],"mid_column":["squid","lanternfish"],"benthic":["deep sea fish"]},
            "note":"stable open ocean — buoyancy dominant"},
        "monsoon":  {"w1":.30,"w2":.30,"w3":.40,"L":1.6,
            "species":{"surface_drift":["seabird","tuna"],"mid_column":["squid"],"benthic":["deep sea fish"]},
            "note":"monsoon surface mixing"},
        "winter":   {"w1":.28,"w2":.30,"w3":.42,"L":1.3,
            "species":{"surface_drift":["seabird"],"mid_column":["lanternfish"],"benthic":["deep sea fish"]},
            "note":"winter ocean — vertical mixing dominant"},
        "migration":{"w1":.30,"w2":.30,"w3":.40,"L":1.8,
            "species":{"surface_drift":["whale shark","sea turtle","seabird"],"mid_column":["tuna","mahi-mahi"],"benthic":["deep sea fish"]},
            "note":"whale shark and turtle migration"},
        "upwelling":{"w1":.35,"w2":.28,"w3":.37,"L":2.0,
            "species":{"surface_drift":["seabird","tuna","sea turtle"],"mid_column":["anchovy","squid"],"benthic":["deep sea fish"]},
            "note":"upwelling — filter-feeding at all levels"},
        "melt":     {"w1":.28,"w2":.30,"w3":.42,"L":1.5,
            "species":{"surface_drift":["seabird","sea turtle"],"mid_column":["squid"],"benthic":["deep sea fish"]},
            "note":"post-melt surface bloom"},
        "ice":      {"w1":.28,"w2":.28,"w3":.44,"L":1.2,
            "species":{"surface_drift":["seabird"],"mid_column":["lanternfish"],"benthic":["deep sea fish"]},
            "note":"ice season — very low biological activity"},
    },
    "coastal_benthic": {
        "spawning": {"w1":.37,"w2":.36,"w3":.27,"L":2.0,
            "species":{"surface_drift":["cormorant","gull","tern"],"mid_column":["herring","anchovy","juvenile cod"],"benthic":["mussel","flatfish","crab"]},
            "note":"coastal spawning — benthic reproduction peak"},
        "dry":      {"w1":.35,"w2":.35,"w3":.30,"L":1.8,
            "species":{"surface_drift":["cormorant","gull"],"mid_column":["herring","anchovy"],"benthic":["mussel","flatfish","crab"]},
            "note":"dry season — stable coastal conditions"},
        "monsoon":  {"w1":.33,"w2":.37,"w3":.30,"L":1.9,
            "species":{"surface_drift":["gull","tern"],"mid_column":["herring"],"benthic":["mussel","flatfish","crab"]},
            "note":"storm events — particle resuspension from benthic layer"},
        "winter":   {"w1":.35,"w2":.35,"w3":.30,"L":1.8,
            "species":{"surface_drift":["cormorant","gull"],"mid_column":["herring","anchovy"],"benthic":["mussel","flatfish","crab"]},
            "note":"coastal benthic active year-round"},
        "migration":{"w1":.33,"w2":.36,"w3":.31,"L":1.9,
            "species":{"surface_drift":["migratory seabird","cormorant"],"mid_column":["herring","sprat"],"benthic":["mussel","crab"]},
            "note":"seabird migration — coastal stopover feeding"},
        "upwelling":{"w1":.38,"w2":.33,"w3":.29,"L":2.1,
            "species":{"surface_drift":["gull","tern","cormorant"],"mid_column":["anchovy","sardine","herring"],"benthic":["mussel","flatfish","crab"]},
            "note":"upwelling event — all trophic levels active"},
    },
    "polar": {
        "melt":     {"w1":.25,"w2":.30,"w3":.45,"L":2.0,
            "species":{"surface_drift":["penguin","albatross","whale"],"mid_column":["krill","Arctic cod"],"benthic":["amphipod","sea spider"]},
            "note":"ice melt — krill bloom, whale migration, highest polar vulnerability"},
        "ice":      {"w1":.27,"w2":.29,"w3":.44,"L":1.3,
            "species":{"surface_drift":["seal","polar bear"],"mid_column":["Arctic cod"],"benthic":["amphipod","sea spider"]},
            "note":"ice season — under-ice particle accumulation"},
        "migration":{"w1":.26,"w2":.29,"w3":.45,"L":1.9,
            "species":{"surface_drift":["penguin","albatross","humpback whale"],"mid_column":["krill","Arctic cod"],"benthic":["amphipod","sea spider"]},
            "note":"whale and penguin migration — particle drift corridors"},
        "spawning": {"w1":.27,"w2":.30,"w3":.43,"L":1.8,
            "species":{"surface_drift":["penguin","albatross"],"mid_column":["krill","capelin"],"benthic":["amphipod","sea spider"]},
            "note":"polar breeding — krill spawning at peak vulnerability"},
    },
}

# ── Sub-score lookup tables ───────────────────────────────────────────────────
PATHWAY = {
    "fiber":    {"label": "gill entanglement",     "P": 9},
    "fragment": {"label": "digestive lodging",     "P": 7},
    "film":     {"label": "external entanglement", "P": 6},
    "pellet":   {"label": "false satiation",       "P": 7},
}

DEPTH_ZONE = {
    "fiber":    "surface_drift",
    "film":     "surface_drift",
    "fragment": "mid_column",
    "pellet":   "benthic",
}

DEPTH_H = {
    "surface_drift": 8.5,
    "mid_column":    6.0,
    "benthic":       5.0,
}

SEVERITY_BANDS = [
    (81, "Critical"),
    (61, "High"),
    (41, "Moderate"),
    (21, "Low"),
    (0,  "Negligible"),
]

# ── Per-particle geometry → sub-scores ───────────────────────────────────────
def compute_S(morph: str, geom: dict) -> float:
    """Toxin load — SA:V proxy from 2D geometry."""
    feret_max = geom["max_feret_um"]
    feret_min = geom["min_feret_um"]
    area      = geom["area_um2"]
    perimeter = geom["perimeter_um"]
    convexity = geom.get("convexity", geom.get("solidity", 0.8))
    rugosity  = geom.get("rugosity", 1.0)

    if morph == "fiber":
        width = max(feret_min, 0.1)
        sav = 4 / (width / 1000)
    elif morph == "pellet":
        d = max(feret_max, 0.1)
        sav = 6 / (d / 1000)
    elif morph == "fragment":
        sav = (perimeter / area) / convexity * rugosity
    else:  # film
        sav = perimeter / area

    size = feret_max
    if size < 100:    modifier = 1.4
    elif size < 1000: modifier = 1.2
    elif size <= 5000: modifier = 1.0
    else:             modifier = 0.7

    raw = sav * modifier
    return min(10.0, round(raw / 80 * 10, 3))


def compute_P(morph: str, geom: dict) -> tuple:
    """Biological pathway — shape → entry point."""
    circularity  = geom["circularity"]
    aspect_ratio = geom["aspect_ratio"]
    feret_max    = geom["max_feret_um"]

    if circularity > 0.85 and feret_max < 100:
        return 10.0, "cellular penetration"
    if aspect_ratio > 10:
        return 9.0, "gill entanglement"
    if circularity < 0.4 and 100 < feret_max < 3000:
        return 7.0, "digestive lodging"
    if feret_max > 2000 and circularity < 0.3:
        return 6.0, "external entanglement"

    p = PATHWAY[morph]
    return float(p["P"]), p["label"]


def compute_H(morph: str, geom: dict) -> tuple:
    """Buoyancy / depth zone from particle physics."""
    aspect_ratio = geom["aspect_ratio"]
    area         = geom["area_um2"]

    if aspect_ratio > 8 or area < 500:
        dz = "surface_drift"
    elif area > 5_000_000:
        dz = "surface_drift"
    else:
        dz = DEPTH_ZONE.get(morph, "mid_column")

    return DEPTH_H[dz], dz


# ── Single particle scorer ────────────────────────────────────────────────────
def score_particle(morph: str, geometry: dict,
                   confidence: float, w1: float, w2: float, w3: float) -> dict:
    S = compute_S(morph, geometry)
    P, pathway_label = compute_P(morph, geometry)
    H, depth_zone    = compute_H(morph, geometry)

    Tp_raw = S * w1 + P * w2 + H * w3
    Tp     = round(Tp_raw * confidence, 4)

    return {
        "S": round(S, 3), "P": P, "H": H,
        "pathway":    pathway_label,
        "depth_zone": depth_zone,
        "Tp":         Tp,
        "Tp_raw":     round(Tp_raw, 4),
    }


# ── Sample-level aggregation ──────────────────────────────────────────────────
def aggregate_sample(scored_particles: list, image_area_cm2: float = 4.0) -> dict:
    n = len(scored_particles)
    if n == 0:
        return {"T_base": 0, "D": 1.0, "diversity": 1.0, "n": 0}

    tps    = sorted([p["Tp"] for p in scored_particles], reverse=True)
    top_k  = max(1, math.ceil(n * 0.2))
    top_k_mean = sum(tps[:top_k]) / top_k
    full_mean  = sum(tps) / n
    T_base = top_k_mean * 0.7 + full_mean * 0.3

    density = n / image_area_cm2
    D = min(2.5, 1 + math.log10(max(1, density)) * 0.6)

    morph_counts = {}
    for p in scored_particles:
        m = p.get("morph", "unknown")
        morph_counts[m] = morph_counts.get(m, 0) + 1

    shannon = -sum((c/n) * math.log(c/n) for c in morph_counts.values() if c > 0)
    diversity_mod = min(1.3, 1 + shannon * 0.15)

    return {
        "T_base":          round(T_base, 4),
        "D":               round(D, 3),
        "diversity":       round(diversity_mod, 3),
        "n":               n,
        "density_per_cm2": round(density, 2),
        "morph_counts":    morph_counts,
    }


# ── Narrative generator ───────────────────────────────────────────────────────
def generate_narrative(T_final: int, band: str, scored_particles: list,
                       zone: str, season: str, ctx: dict, agg: dict) -> str:
    pathways = [p["pathway"] for p in scored_particles]
    dominant_pathway = max(set(pathways), key=pathways.count)
    avg_S = sum(p["S"] for p in scored_particles) / len(scored_particles)
    avg_P = sum(p["P"] for p in scored_particles) / len(scored_particles)
    avg_H = sum(p["H"] for p in scored_particles) / len(scored_particles)

    w1, w2, w3 = ctx["w1"], ctx["w2"], ctx["w3"]
    driver = ("toxin load (S)" if avg_S*w1 >= avg_P*w2 and avg_S*w1 >= avg_H*w3
              else "biological pathway (P)" if avg_P*w2 >= avg_H*w3
              else "depth zone exposure (H)")

    morph_counts = agg["morph_counts"]
    dom_morph = max(morph_counts, key=morph_counts.get)
    dom_frac  = round(morph_counts[dom_morph] / agg["n"] * 100)

    active_species = list({s for dz in ctx["species"].values() for s in dz})

    narr = (
        f"This sample scores {T_final}/100 — {band}. "
        f"Primary driver is {driver}, evaluated through the lens of {zone.replace('_',' ')} "
        f"during {season} season. "
        f"{dom_frac}% of particles are {dom_morph}s, entering organisms via {dominant_pathway}. "
    )
    if avg_S > 6:
        narr += "Toxin load is elevated — high SA:V particles are concentrating persistent organic pollutants. "
    if avg_P >= 8:
        narr += "Pathway severity is acute — gill entanglement is the dominant documented harm mechanism. "
    narr += (
        f"Species at active risk: {', '.join(active_species)}. "
        f"Context: {ctx['note']}."
    )
    return narr


# ── Master pipeline function ──────────────────────────────────────────────────
def compute_threat_score(
    particles:      list,
    zone:           str,
    season:         str,
    image_area_cm2: float = 4.0,
) -> dict:
    """
    Main entry point.

    particles = [
        {
            "morph":      "fiber",
            "confidence": 0.91,
            "geometry": {
                "max_feret_um": 87.4,  "min_feret_um": 4.1,
                "area_um2":     412.5, "perimeter_um": 198.3,
                "circularity":  0.12,  "aspect_ratio": 21.3,
                "convexity":    0.71,  "rugosity":     1.18,
            }
        },
        ...
    ]
    """
    ctx = LAYER0[zone][season]
    w1, w2, w3 = ctx["w1"], ctx["w2"], ctx["w3"]

    scored = []
    for p in particles:
        result = score_particle(p["morph"], p["geometry"], p["confidence"], w1, w2, w3)
        result["morph"] = p["morph"]
        scored.append(result)

    agg = aggregate_sample(scored, image_area_cm2)

    T_raw   = agg["T_base"] * agg["D"] * agg["diversity"] * ctx["L"]
    T_final = min(100, round(T_raw / 35 * 100))
    band    = next(b for thresh, b in SEVERITY_BANDS if T_final >= thresh)

    narrative = generate_narrative(T_final, band, scored, zone, season, ctx, agg)

    return {
        "T_final":        T_final,
        "band":           band,
        "narrative":      narrative,
        "sub_scores": {
            "avg_S": round(sum(p["S"] for p in scored) / len(scored), 3),
            "avg_P": round(sum(p["P"] for p in scored) / len(scored), 3),
            "avg_H": round(sum(p["H"] for p in scored) / len(scored), 3),
        },
        "aggregation":    agg,
        "weights":        {"w1": w1, "w2": w2, "w3": w3, "L": ctx["L"]},
        "particles":      scored,
        "species_at_risk": list({s for dz in ctx["species"].values() for s in dz}),
    }
