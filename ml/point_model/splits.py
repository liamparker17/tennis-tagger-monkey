import hashlib

HOLDOUT_FRACTION = 0.10

def is_holdout(match_name: str, fraction: float = HOLDOUT_FRACTION) -> bool:
    h = int(hashlib.sha1(match_name.encode()).hexdigest()[:8], 16)
    return (h % 1000) / 1000.0 < fraction

def split_matches(names: list[str]) -> tuple[list[str], list[str]]:
    train = [n for n in names if not is_holdout(n)]
    val = [n for n in names if is_holdout(n)]
    return train, val
