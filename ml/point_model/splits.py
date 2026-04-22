import hashlib

HOLDOUT_FRACTION = 0.10
CLIP_HOLDOUT_FRACTION = 0.20

def is_holdout(match_name: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    h = int(hashlib.sha1(match_name.encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0 < fraction

def split_matches(names: list[str]) -> tuple[list[str], list[str]]:
    train = [n for n in names if not is_holdout(n)]
    val = [n for n in names if is_holdout(n)]
    # Fallback: if hash-holdout left val empty but we have >=2 matches,
    # promote the last train match to val so we actually evaluate.
    if not val and len(train) >= 2:
        val = [train.pop()]
    return train, val


def split_clips(match_name: str, clip_stems: list[str],
                fraction: float = CLIP_HOLDOUT_FRACTION
                ) -> tuple[set[str], set[str]]:
    """Single-match fallback: hash each clip stem and hold out a deterministic
    ~fraction of them as val. Used when the whole dataset is one match so
    split_matches() returns no val — without this we train blind."""
    train: set[str] = set()
    val: set[str] = set()
    for stem in clip_stems:
        h = int(hashlib.sha1(f"{match_name}/{stem}".encode()).hexdigest()[:8], 16)
        (val if (h % 1000) / 1000.0 < fraction else train).add(stem)
    # Guarantee both sides are non-empty. With very few clips the hash can
    # put everything on one side; pull the first/last stem across the line.
    if clip_stems and not val:
        val = {clip_stems[-1]}; train.discard(clip_stems[-1])
    if clip_stems and not train:
        train = {clip_stems[0]}; val.discard(clip_stems[0])
    return train, val
