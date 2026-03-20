import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from image_preprocessing import preprocess_face

# MODELLER
face_model = YOLO("models/face_hiding_best.pt")
hat_model  = YOLO("models/face_concealment_hat_best.pt")

cap = cv2.VideoCapture("test_videos/atm_pexels_hood.mp4")

# ==============================
# STABILIZATION + TIME TRACKING
# ==============================
label_history = defaultdict(lambda: deque(maxlen=7))
presence_counter = defaultdict(int)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    persons = []

    face_results = face_model(frame, conf=0.4, verbose=False)[0]

    # --------------------------
    # PERSON OLUŞTUR
    # --------------------------
    for box in face_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        face_img = frame[y1:y2, x1:x2]
        face_img = preprocess_face(face_img)
        if face_img is None:
            continue

        hat_results = hat_model(face_img, conf=0.4, verbose=False)[0]

        labels = ["face"]

        if len(hat_results.boxes) > 0:
            best = max(hat_results.boxes, key=lambda b: float(b.conf[0]))
            cls = int(best.cls[0])
            labels.append(hat_model.names[cls])

        person_id = (x1, y1, x2, y2)
        persons.append({
            "id": person_id,
            "bbox": (x1, y1, x2, y2),
            "labels": labels
        })

    # --------------------------
    # RISK HESABI (MAIN.PY LOGIC)
    # --------------------------
    for idx, p in enumerate(persons):
        bx1, by1, bx2, by2 = p["bbox"]
        labels = set(p["labels"])

        # STABILIZATION
        for l in labels:
            if l != "face":
                label_history[p["id"]].append(l)

        stable_label = None
        if label_history[p["id"]]:
            stable_label = max(
                set(label_history[p["id"]]),
                key=label_history[p["id"]].count
            )

        # TIME TRACKING
        presence_counter[p["id"]] += 1
        time_visible = presence_counter[p["id"]] / 30  # saniye (≈30fps)

        risk = 0

        # BASE RISK
        if stable_label == "balaclava":
            risk = 70
        elif stable_label in ["medicine mask", "scarf"]:
            risk = 25
        elif stable_label in ["hat", "hood", "cap"]:
            risk = 25

        # ⏱️ SÜRE ETKİSİ
        if time_visible > 3:
            risk += 10
        if time_visible > 6:
            risk += 15

        # 👥 KİŞİ SAYISI ETKİSİ
        if len(persons) >= 2:
            risk += 10

        risk = min(risk, 100)

        # --------------------------
        # DRAW
        # --------------------------
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

    cv2.imshow("ATM RISK TEST VIDEO", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

