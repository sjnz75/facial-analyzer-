def diagnose_median_line(deviation_mm):
    if deviation_mm > 2:
        return "Deviazione significativa"
    if 0.5 <= deviation_mm <= 1:
        return "Clinicamente accettabile"
    return "Normale"


def diagnose_angle_line(angle_deg, line_type):
    thresholds = {
        "interpupillare": 1.5,
        "commissurale": 1.75,
        "interalare": 0  # usiamo mm per ali
    }
    th = thresholds.get(line_type, 2)
    if angle_deg > th:
        if line_type == "interpupillare": return "Asimmetria evidente"
        if line_type == "commissurale": return "Sorriso asimmetrico"
    return "Normale"


def diagnose_thirds(percentages):
    comments = []
    for pct in percentages:
        if pct < 30:
            comments.append("Ipotrofia")
        elif pct > 36:
            comments.append("Ipertrofia")
        else:
            comments.append("Normale")
    return comments
