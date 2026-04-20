STROKE_CLASSES = ["Forehand","Backhand","Serve","Volley","Slice","Smash","DropShot","Lob","Other"]
OUTCOME_CLASSES = ["Ace","DoubleFault","Winner","ForcedError","UnforcedError"]
STROKE_TO_IDX = {s: i for i, s in enumerate(STROKE_CLASSES)}
OUTCOME_TO_IDX = {s: i for i, s in enumerate(OUTCOME_CLASSES)}
STROKE_PAD_INDEX = -100
NUM_STROKE = len(STROKE_CLASSES)
NUM_OUTCOME = len(OUTCOME_CLASSES)

def stroke_index(name: str) -> int:
    return STROKE_TO_IDX.get(name.strip(), STROKE_TO_IDX["Other"])

def outcome_index(label: str) -> int:
    return OUTCOME_TO_IDX.get(label.strip(), OUTCOME_TO_IDX["UnforcedError"])
