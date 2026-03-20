import cv2
from ultralytics import YOLO
from image_preprocessing import preprocess_face

# ==============================
# MODELLER
# ==============================
face_model = YOLO("models/face_hiding_best.pt")
hat_model  = YOLO("models/face_concealment_hat_best.pt")

VIDEO_PATH = "test_videos/atm_pexels.mp4"  # ← test ettiğin video buysa

# ==============================
# RISK HESABI (VİDEO)
# ==============================
def calculate_risk(labels, person_count):
    risk = 0

    if "balaclava" in labels:
        risk = max(risk, 70)

    if "medicine mask" in labels or "scarf" in labels:
        risk += 25

    if "hat" in labels or "hood" in labels or "cap" in labels:
        risk += 25

    if person_count > 1:
        risk += 10

    return min(risk, 100)

# ==============================
# VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video açılamadı")
    exit()

print("✅ Video risk testi başladı | Çıkış: Q")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_results = face_model(frame, conf=0.4, verbose=False)[0]
    persons = []

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
            labels.append(hat_model.names[int(best.cls[0])])

        persons.append({
            "bbox": (x1, y1, x2, y2),
            "labels": labels
        })

    for p in persons:
        risk = calculate_risk(p["labels"], len(persons))
        x1, y1, x2, y2 = p["bbox"]

        color = (0, 255, 0)
        if risk >= 80:
            color = (0, 0, 255)
        elif risk >= 50:
            color = (0, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"RISK: {risk}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("VIDEO RISK TEST", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

