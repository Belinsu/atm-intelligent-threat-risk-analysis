import cv2
import time
import yaml

from ultralytics import YOLO
from collections import defaultdict, deque

from utils.weapon_detector import WeaponDetector
from utils.risk_engine import weapon_risk_score
from image_preprocessing import preprocess_face


# ==============================
# MODELLER
# ==============================
face_model = YOLO("models/face_hiding_best.pt")
hat_model  = YOLO("models/face_concealment_hat_best.pt")

with open("weapon_config.yaml") as f:
    weapon_cfg = yaml.safe_load(f)

weapon_detector = WeaponDetector(weapon_cfg)


# ==============================
# KAMERA
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera açılamadı")
    exit()

print("✅ ATM Risk AI çalışıyor | Çıkış: Q")


# ==============================
# LABEL STABILIZATION
# ==============================
label_history = defaultdict(lambda: deque(maxlen=7))


# ==============================
# TIME TRACKING (YENİ)
# ==============================
person_time = defaultdict(lambda: {
    "first_seen": time.time(),
    "last_seen": time.time()
})


# ==============================
# ANA DÖNGÜ
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons = []

    # --------------------------
    # FACE DETECTION
    # --------------------------
    face_results = face_model(frame, conf=0.4, verbose=False)[0]

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

        # --------------------------
        # FACE CROP (PADDING)
        # --------------------------
        h, w, _ = frame.shape

        pad_x = int((x2 - x1) * 0.4)
        pad_y = int((y2 - y1) * 0.5)

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w, x2 + pad_x)
        y2p = min(h, y2 + pad_y)

        face_img = frame[y1p:y2p, x1p:x2p]

        # --------------------------
        # PREPROCESS
        # --------------------------
        face_img = preprocess_face(face_img)
        if face_img is None:
            continue

        # --------------------------
        # HAT / MASK MODEL
        # --------------------------
        hat_results = hat_model(face_img, conf=0.4, verbose=False)[0]

        labels = ["face"]
        hat_label = None

        if len(hat_results.boxes) > 0:
            best_box = max(hat_results.boxes, key=lambda b: float(b.conf[0]))
            hat_cls = int(best_box.cls[0])
            hat_label = hat_model.names[hat_cls]

        if hat_label:
            labels.append(hat_label)

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
    # KİŞİ BAZLI RİSK
    # --------------------------
    for idx, p in enumerate(persons):
        labels = set(p["labels"])
        bx1, by1, bx2, by2 = p["bbox"]
        pid = p["id"]

        risk = 0

        # 🔴 WEAPON (OVERRIDE)
        weapon_dets = weapon_detector.detect(frame)
        if weapon_risk_score(weapon_dets) > 0:
            risk = 100

        # 🟥 BALACLAVA
        if "balaclava" in labels:
            risk = max(risk, 70)

        # 🟡 MASK / SCARF
        if "medicine mask" in labels or "scarf" in labels:
            risk += 25

        # 🟡 HAT / HOOD / CAP
        if "hat" in labels or "hood" in labels or "cap" in labels:
            risk += 25

        # 🟡 HOOD + MASK COMBO
        if ("hood" in labels or "cap" in labels) and (
            "medicine mask" in labels or "scarf" in labels
        ):
            risk = max(risk, 55)

        # ✋ HAND
        if p["hand"]:
            risk += 15

        # ==========================
        # ⏱️ SÜREYE BAĞLI RİSK (YENİ)
        # ==========================
        now = time.time()

        if pid not in person_time:
            person_time[pid]["first_seen"] = now

        person_time[pid]["last_seen"] = now
        waiting_time = now - person_time[pid]["first_seen"]

        if waiting_time > 30:
            extra_time = waiting_time - 30
            risk += int(extra_time // 10) * 5

        # ==========================
        # 👥 KİŞİ SAYISI RİSKİ (YENİ)
        # ==========================
        if person_count > 2:
            risk += (person_count - 2) * 10

        # 🔒 LIMIT
        risk = min(risk, 100)

        # --------------------------
        # ÇİZİM
        # --------------------------
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

    cv2.imshow("ATM Risk AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

