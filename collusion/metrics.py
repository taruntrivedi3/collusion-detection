"""Scoring detected coalitions against planted ones."""
import numpy as np


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def score(detected, truth):
    """detected: list of frozensets in detection order (peeling order for the scan);
    truth: list of planted frozensets.

    detected        : any coalition reported
    exact           : reported set of coalitions equals the planted set exactly
    exact_first     : first reported coalition is a planted coalition (NaN if none planted)
    recovered       : fraction of planted coalitions reported exactly (NaN if none planted)
    best_jacc       : mean over planted of best Jaccard with any detection (0 if none detected)
    false_coal      : # reported sets that are not exactly a planted coalition
    false_disjoint  : # reported sets that overlap no planted coalition
    """
    det, tru = [frozenset(d) for d in detected], {frozenset(t) for t in truth}
    dset = set(det)
    nan = float("nan")
    best = [max((jaccard(t, d) for d in det), default=0.0) for t in tru]
    return {
        "detected": bool(det),
        "exact": dset == tru,
        "exact_first": float(bool(det) and det[0] in tru) if tru else nan,
        "recovered": sum(t in dset for t in tru) / len(tru) if tru else nan,
        "best_jacc": sum(best) / len(best) if best else nan,
        "false_coal": sum(d not in tru for d in det),
        "false_disjoint": sum(not any(d & t for t in tru) for d in det),
    }


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n trials -> (lo, hi) as fractions."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half
