import cv2
import time
import yaml
import os
import numpy as np
import urllib.request

from ultralytics import YOLO
from collections import defaultdict, deque

from utils.weapon_detector import WeaponDetector
from utils.risk_engine import weapon_risk_score
from image_preprocessing import preprocess_face

# ==============================
# TEST URLS
# ==============================
TEST_URLS = [
    "https://www.readingeagle.com/wp-content/uploads/migration/2015/05/7815940a9ad0ad17bc7960b6d704825d.jpg",
    "https://c8.alamy.com/comp/C37JHD/police-exercise-bank-robbery-bank-robber-with-a-gun-takes-hostages-C37JHD.jpg"
]

def load_image_from_url(url):
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )
        resp = urllib.request.urlopen(req, timeout=10)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"❌ URL okunamadı: {url} | {e}")
        return None

# ==============================
# URL IMAGE TEST
# ==============================
for url in TEST_URLS:
    frame = load_image_from_url(url)
    if frame is None:
        continue

    persons = []

    # ⬇️ BURADAN SONRASI
    # SENİN MEVCUT IMAGE TEST PIPELINE’IN
    # TEK SATIR DEĞİŞMİYOR


# ==============================
# MODELLER
# ==============================
face_model = YOLO("models/face_hiding_best.pt")
hat_model  = YOLO("models/face_concealment_hat_best.pt")

with open("weapon_config.yaml") as f:
    weapon_cfg = yaml.safe_load(f)

weapon_detector = WeaponDetector(weapon_cfg)


# ==============================
# LABEL STABILIZATION
# ==============================
label_history = defaultdict(lambda: deque(maxlen=7))


# ==============================
# TIME TRACKING
# ==============================
person_time = defaultdict(lambda: {
    "first_seen": time.time(),
    "last_seen": time.time()
})


# ==============================
# IMAGE SOURCE
# ==============================
IMAGE_DIR = "test_images"


# ==============================
# ANA DÖNGÜ (IMAGE TEST)
# ==============================
for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    frame = cv2.imread(os.path.join(IMAGE_DIR, img_name))
    if frame is None:
        continue

    persons = []

    # --------------------------
    # FACE DETECTION (PREPROCESS VAR)
    # --------------------------
    pre_frame = preprocess_face(frame)
    if pre_frame is None:
        pre_frame = frame

    face_results = face_model(pre_frame, conf=0.4, verbose=False)[0]

    if len(face_results.boxes) == 0:
        cv2.putText(
            frame,
            "FACE NOT VISIBLE | POSSIBLE CONCEALMENT",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # --------------------------
    # FACE LOOP
    # --------------------------
    for box in face_results.boxes:
        cls = int(box.cls[0])
        name = face_model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        h, w, _ = frame.shape

        pad_x = int((x2 - x1) * 0.4)
        pad_y = int((y2 - y1) * 0.5)

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w, x2 + pad_x)
        y2p = min(h, y2 + pad_y)

        # 🔴 RAW FACE CROP (PREPROCESS YOK)
        face_crop = frame[y1p:y2p, x1p:x2p]

        # --------------------------
        # HAT / MASK MODEL (RAW)
        # --------------------------
        hat_results = hat_model(face_crop, conf=0.4, verbose=False)[0]

        labels = ["face"]
        if len(hat_results.boxes) > 0:
            for b in hat_results.boxes:
                labels.append(hat_model.names[int(b.cls[0])])

        # --------------------------
        # STABILIZATION
        # --------------------------
        person_id = (x1, y1, x2, y2)
        detail_label = next((l for l in labels if l != "face"), None)

        if detail_label:
            label_history[person_id].append(detail_label)

        stable_label = None
        if label_history[person_id]:
            stable_label = max(
                set(label_history[person_id]),
                key=label_history[person_id].count
            )

        final_labels = ["face"]
        if stable_label:
            final_labels.append(stable_label)

        persons.append({
            "id": person_id,
            "bbox": (x1, y1, x2, y2),
            "labels": final_labels,
            "hand": name == "hand"
        })

    # 👥 KİŞİ SAYISI
    person_count = len(persons)

    # --------------------------
    # RISK (❗ BİREBİR SENİN LOGIC)
    # --------------------------
    for idx, p in enumerate(persons):
        labels = set(p["labels"])
        bx1, by1, bx2, by2 = p["bbox"]
        pid = p["id"]

        risk = 0

        weapon_dets = weapon_detector.detect(frame)
        if weapon_risk_score(weapon_dets) > 0:
            risk = 100

        if "balaclava" in labels:
            risk = max(risk, 70)

        if "medicine mask" in labels or "scarf" in labels:
            risk += 25

        if "hat" in labels or "hood" in labels or "cap" in labels:
            risk += 25

        if ("hood" in labels or "cap" in labels) and (
            "medicine mask" in labels or "scarf" in labels
        ):
            risk = max(risk, 55)

        if p["hand"]:
            risk += 15

        now = time.time()
        waiting = now - person_time[pid]["first_seen"]
        if waiting > 30:
            risk += int((waiting - 30) // 10) * 5

        if person_count > 2:
            risk += (person_count - 2) * 10

        risk = min(risk, 100)

        color = (0, 255, 0)
        if risk >= 80:
            color = (0, 0, 255)
        elif risk >= 50:
            color = (0, 255, 255)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            frame,
            f"Person {idx+1} | RISK: {risk}%",
            (bx1, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("ATM Risk AI | IMAGE TEST", frame)
    cv2.waitKey(0)   # her görselde dur

cv2.destroyAllWindows()

