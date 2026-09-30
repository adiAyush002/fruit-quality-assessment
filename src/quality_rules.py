"""Rule-based quality assessment. NOT a deep-learning prediction.

Edit QUALITY_TABLE to change the rules. Keys are (defect, ripeness).
"""

LOW_CONFIDENCE_THRESHOLD = 0.60

QUALITY_TABLE = {
    ("healthy", "ripe"): "Good",
    ("healthy", "unripe"): "Average",
    ("healthy", "overripe"): "Average",
    ("bruised", "ripe"): "Average",
    ("bruised", "unripe"): "Average",
    ("bruised", "overripe"): "Poor",
    ("spotted", "ripe"): "Average",
    ("spotted", "unripe"): "Average",
    ("spotted", "overripe"): "Poor",
    ("rotten", "ripe"): "Poor",
    ("rotten", "unripe"): "Poor",
    ("rotten", "overripe"): "Poor",
}

DEFAULT_QUALITY = "Unknown"


def assess_quality(defect, ripeness, confidences=None):
    quality = QUALITY_TABLE.get((defect, ripeness), DEFAULT_QUALITY)
    explanation = (f"The defect model reports '{defect}' and the ripeness model reports '{ripeness}'. "
                   f"Under the rule table in src/quality_rules.py this combination is rated {quality}.")
    warnings = []
    for task, conf in (confidences or {}).items():
        if conf < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Low confidence for {task} ({conf * 100:.0f}%). "
                            "The result may be unreliable - try a clearer photo.")
    return {"quality": quality, "explanation": explanation, "warnings": warnings}
