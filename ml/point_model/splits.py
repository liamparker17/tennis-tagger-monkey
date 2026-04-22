import hashlib

HOLDOUT_FRACTION = 0.10

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
