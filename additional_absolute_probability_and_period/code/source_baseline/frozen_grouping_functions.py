"""Exact isolated functions from frozen original source; no entrypoint side effects."""
import numpy as np
import pandas as pd

def collapse_icu_type(value: object) -> str:
    """Collapse granular MIMIC care units into stable clinical groups."""
    text = str(value or "").lower()
    if "cardiac" in text or "coronary" in text or "cvicu" in text or "ccu" in text:
        return "cardiac"
    if "neuro" in text:
        return "neuro"
    if "medical/surgical" in text or "micu/sicu" in text:
        return "mixed"
    if "surgical" in text or "surgery" in text or "trauma" in text or "sicu" in text:
        return "surgical_trauma"
    if "medical" in text or "micu" in text:
        return "medical"
    # Very small residual units have no outcome events in this locked cohort;
    # combine them with the medical reference group to avoid separation.
    return "medical"

def collapse_admission_type(value: object) -> str:
    """Collapse MIMIC admission types into emergency, urgent, planned, or observation."""
    text = str(value or "").upper()
    if "OBSERV" in text:
        return "observation"
    if "URGENT" in text:
        return "urgent"
    if "ELECTIVE" in text or "SURGICAL SAME DAY" in text:
        return "planned"
    if "EMER" in text:
        return "emergency"
    return "other"
