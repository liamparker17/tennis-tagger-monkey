STROKE_CLASSES = ["Forehand","Backhand","Serve","Volley","Slice","Smash","DropShot","Lob","Other"]
OUTCOME_CLASSES = ["Ace","DoubleFault","Winner","ForcedError","UnforcedError"]
STROKE_TO_IDX = {s: i for i, s in enumerate(STROKE_CLASSES)}
OUTCOME_TO_IDX = {s: i for i, s in enumerate(OUTCOME_CLASSES)}
STROKE_PAD_INDEX = -100
NUM_STROKE = len(STROKE_CLASSES)
NUM_OUTCOME = len(OUTCOME_CLASSES)


def stroke_index(name: str) -> int:
    """Map a raw Dartfish stroke string to a canonical stroke class index.

    Dartfish emits strings like "1st Serve Made", "Forehand Return Made",
    "Backhand Volley", "Forehand Slice", etc. We check the more specific
    modifiers (Volley/Slice/Smash/Lob/DropShot/Serve) before the wing
    (Forehand/Backhand), since a "Forehand Volley" is primarily a Volley.
    """
    s = name.strip().lower()
    if not s: return STROKE_TO_IDX["Other"]
    if "serve" in s: return STROKE_TO_IDX["Serve"]
    if "smash" in s or "overhead" in s: return STROKE_TO_IDX["Smash"]
    if "volley" in s: return STROKE_TO_IDX["Volley"]
    if "lob" in s: return STROKE_TO_IDX["Lob"]
    if "drop" in s: return STROKE_TO_IDX["DropShot"]
    if "slice" in s: return STROKE_TO_IDX["Slice"]
    if "forehand" in s: return STROKE_TO_IDX["Forehand"]
    if "backhand" in s: return STROKE_TO_IDX["Backhand"]
    return STROKE_TO_IDX["Other"]


def outcome_index(label: str) -> int:
    """parse.py already canonicalises this to one of OUTCOME_CLASSES."""
    return OUTCOME_TO_IDX.get(label.strip(), OUTCOME_TO_IDX["UnforcedError"])
