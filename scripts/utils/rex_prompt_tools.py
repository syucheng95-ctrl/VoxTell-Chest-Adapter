import re
from dataclasses import dataclass, asdict
from typing import Optional


LESION_PATTERNS = [
    ("ground-glass opacity", r"\bground(?:-|\s)?glass opacity\b|\bground(?:-|\s)?glass opacities\b|\bground(?:-|\s)?glass densit(?:y|ies)\b|\bground(?:-|\s)?glass appearance\b"),
    ("consolidation", r"\bconsolidation\b|\bconsolidations\b|\bconsolidative\b|\bconsolidative density\b"),
    ("nodule", r"\bnodule\b|\bnodules\b"),
    ("atelectasis", r"\batelectasis\b|\batelectatic\b"),
    ("emphysema", r"\bemphysema\b|\bemphysematous\b"),
    ("bronchiectasis", r"\bbronchiectasis\b|\bbronchiectatic\b"),
    ("pleural effusion", r"\bpleural effusion\b|\beffusion\b"),
    ("thickening", r"\bthickening\b"),
    ("scarring", r"\bscarring\b"),
    ("fibrosis", r"\bfibrosis\b|\bfibrotic\b"),
    ("infiltration", r"\binfiltration\b|\binfiltrations\b"),
    ("mass", r"\bmass\b|\bmasses\b"),
    ("cyst", r"\bcyst\b|\bcysts\b"),
    ("opacity", r"\bopacity\b|\bopacities\b"),
]

MORPHOLOGY_TERMS = [
    "subpleural",
    "linear",
    "nodular",
    "calcified",
    "calcific",
    "patchy",
    "peripheral",
    "fibrotic",
    "peribronchial",
    "tree-in-bud",
    "mosaic",
    "centriacinar",
    "centrilobular",
    "paraseptal",
    "reticulonodular",
    "irregular",
    "spiculated",
    "cavitary",
    "solid",
    "nonspecific",
]

EXTENT_TERMS = [
    "focal",
    "diffuse",
    "minimal",
    "multiple",
    "several",
    "widespread",
    "multifocal",
    "occasional",
    "few",
]

LATERALITY_TERMS = ["bilateral", "both", "right", "left"]

ANATOMY_PATTERNS = [
    ("right upper lobe", r"\bright upper lobe\b|\bupper lobe of the right lung\b"),
    ("right middle lobe", r"\bright middle lobe\b|\bmiddle lobe of the right lung\b"),
    ("right lower lobe", r"\bright lower lobe\b|\blower lobe of the right lung\b"),
    ("left upper lobe", r"\bleft upper lobe\b|\bupper lobe of the left lung\b"),
    ("left lower lobe", r"\bleft lower lobe\b|\blower lobe of the left lung\b"),
    ("right lung", r"\bright lung\b"),
    ("left lung", r"\bleft lung\b"),
    ("lingular segment", r"\blingular segment\b|\blingular segments\b"),
    ("lingula", r"\blingula\b|\binferior lingula\b"),
    ("lung apices", r"\blung apices\b|\bapices\b"),
    ("both lungs", r"\bboth lungs\b|\bboth lung\b|\bbilateral lungs\b"),
    ("lungs", r"\blungs\b|\blung parenchyma\b"),
]

LOCATION_TERMS = [
    "posterobasal segment",
    "posterobasal level",
    "laterobasal segment",
    "laterobasal level",
    "anterobasal segment",
    "mediobasal segment",
    "superior segment",
    "anterior segment",
    "posterior segment",
    "medial segment",
    "lateral segment",
    "apical segment",
    "basal segment",
    "subsegmental",
    "subpleural",
    "peripheral",
    "central",
    "adjacent to the diaphragm",
    "inferior lingula",
    "paramediastinal",
    "at the level of the left lung lower lobe fissure",
]

SIZE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?\s*mm\b",
    r"\b\d+(?:\.\d+)?\s*mm\b",
    r"\bsubcentimeter\b",
]


@dataclass
class StructuredPrompt:
    raw_text: str
    lesion: Optional[str] = None
    laterality: Optional[str] = None
    anatomy: Optional[str] = None
    location: Optional[str] = None
    morphology: Optional[str] = None
    extent: Optional[str] = None
    size: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _find_first_pattern(text: str, patterns: list[tuple[str, str]]) -> Optional[str]:
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return None


def _find_all_terms(text: str, terms: list[str]) -> list[str]:
    hits = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text):
            hits.append(term)
    return hits


def _choose_laterality(text: str) -> Optional[str]:
    if re.search(r"\bbilateral\b|\bboth\b", text):
        return "bilateral"
    if re.search(r"\bright\b", text):
        return "right"
    if re.search(r"\bleft\b", text):
        return "left"
    return None


def _choose_size(text: str) -> Optional[str]:
    for pattern in SIZE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _pick_location(text: str) -> Optional[str]:
    hits = _find_all_terms(text, LOCATION_TERMS)
    if not hits:
        return None
    # Prefer the most specific multi-word span.
    hits = sorted(hits, key=lambda item: (-len(item), item))
    return hits[0]


def _pick_morphology(text: str, location: Optional[str]) -> Optional[str]:
    hits = _find_all_terms(text, MORPHOLOGY_TERMS)
    if location and location in hits:
        hits = [item for item in hits if item != location]
    if not hits:
        return None
    return ", ".join(sorted(hits))


def _pick_extent(text: str) -> Optional[str]:
    hits = _find_all_terms(text, EXTENT_TERMS)
    if not hits:
        return None
    priority = ["diffuse", "widespread", "multifocal", "multiple", "several", "few", "focal", "minimal", "occasional"]
    for item in priority:
        if item in hits:
            return item
    return hits[0]


def parse_finding(raw_text: str) -> StructuredPrompt:
    text = _normalize_spaces(raw_text.lower())

    lesion = _find_first_pattern(text, LESION_PATTERNS)
    laterality = _choose_laterality(text)
    anatomy = _find_first_pattern(text, ANATOMY_PATTERNS)
    location = _pick_location(text)
    morphology = _pick_morphology(text, location)
    extent = _pick_extent(text)
    size = _choose_size(text)

    return StructuredPrompt(
        raw_text=raw_text,
        lesion=lesion,
        laterality=laterality,
        anatomy=anatomy,
        location=location,
        morphology=morphology,
        extent=extent,
        size=size,
    )


def format_structured_prompt(parsed: StructuredPrompt, style: str = "templated") -> str:
    fields = [
        ("Lesion", parsed.lesion),
        ("Laterality", parsed.laterality),
        ("Anatomy", parsed.anatomy),
        ("Location", parsed.location),
        ("Morphology", parsed.morphology),
        ("Extent", parsed.extent),
        ("Size", parsed.size),
    ]
    populated = [(key, value) for key, value in fields if value]

    if style == "compact":
        return "; ".join(f"{key.lower()}={value}" for key, value in populated)

    return " ".join(f"{key}: {value}." for key, value in populated)
