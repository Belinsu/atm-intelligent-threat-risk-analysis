import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from utils.image_preprocess import adaptive_preprocess
from utils.weapon_detector import WeaponDetector
from utils.risk_engine import weapon_risk_score
import yaml

# ==============================
# AYAR
# ==============================
IMAGE_PATH = "test_images/atm_skimmer.jpg"  # <-- BURAYI DEĞİŞTİR

# ==============================
# MODELLER
# ==============================
face_model = YOLO("models/face_hiding_best.pt")
hat_model  = YOLO("models/face_concealment_hat_best.pt")

with open("weapon_config.yaml") as f:
    weapon_cfg = yaml.safe_load(f)

weapon_detector = WeaponDetector(weapon_cfg)

# ==============================
# IOU
# ==============================
def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# ==============================
# GÖRÜNTÜ YÜKLE
# ==============================
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("❌ Görüntü yüklenemedi:", IMAGE_PATH)
    exit()

frame = adaptive_preprocess(frame)

# ==============================
# ALGILAMA
# ==============================
face_results = face_model(frame, conf=0.4, verbose=False)[0]
hat_results  = hat_model(frame,  conf=0.4, verbose=False)[0]

persons = []

# ------------------------------
# FACE PERSONS
# ------------------------------
for box in face_results.boxes:
    cls = int(box.cls[0])
    name = face_model.names[cls]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    persons.append({
        "bbox": (x1, y1, x2, y2),
        "labels": [name],
        "hand": name == "hand"
    })

# ------------------------------
# HAT → PERSON
# ------------------------------
for box in hat_results.boxes:
    cls = int(box.cls[0])
    name = hat_model.names[cls]

    if name not in ["hat", "hood", "cap"]:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    hat_box = (x1, y1, x2, y2)

    for p in persons:
        if iou(p["bbox"], hat_box) > 0.2:
            p["labels"].append("hat")

# ==============================
# WEAPON
# ==============================
weapon_dets = weapon_detector.detect(frame)
weapon_risk = weapon_risk_score(weapon_dets)

# ==============================
# RİSK + ÇİZİM
# ==============================
for idx, p in enumerate(persons):
    labels = set(p["labels"])
    bx1, by1, bx2, by2 = p["bbox"]

    risk = 0

    if weapon_risk > 0:
        risk = 100

    if "balaclava" in labels:
        risk = max(risk, 70)

    if "medicine mask" in labels or "scarf" in labels:
        risk += 25

    if "hat" in labels:
        risk += 25

    if p["hand"]:
        risk += 15

    risk = min(risk, 100)

    color = (0,255,0)
    if risk >= 80:
        color = (0,0,255)
    elif risk >= 50:
        color = (0,255,255)

    cv2.rectangle(frame, (bx1,by1), (bx2,by2), color, 2)
    cv2.putText(
        frame,
        f"Person {idx+1} | RISK: {risk}%",
        (bx1, by1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# ==============================
# GÖSTER
# ==============================
cv2.imshow("TEST IMAGE", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

