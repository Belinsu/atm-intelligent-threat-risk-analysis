def weapon_risk_score(detections):
    score = 0
    for d in detections:
        if d["label"] == "gun":
            score += 5
        elif d["label"] == "knife":
            score += 3
    return score
