import cv2
import numpy as np

# ------------------------
# TEMEL TEMİZLEME
# ------------------------
def preprocess_frame(frame):
    # Hafif gürültü azaltma
    denoised = cv2.GaussianBlur(frame, (5, 5), 0)

    # Kontrast artırma (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Hafif keskinleştirme
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(contrast, -1, kernel)

    return sharp


# ------------------------
# AŞIRI AYDINLIK DÜZELTME
# ------------------------
def fix_overexposed(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.normalize(v, None, 50, 200, cv2.NORM_MINMAX)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ------------------------
# ORTAM ANALİZİ
# ------------------------
def adaptive_preprocess(frame):
    mean = frame.mean()
    std = frame.std()

    # 🌙 Karanlık
    if mean < 60:
        return preprocess_frame(frame)

    # ☀️ Aşırı aydınlık
    if mean > 180:
        return fix_overexposed(frame)

    # 🌧️ Yağmur / yoğun parazit
    if std > 70:
        return cv2.medianBlur(frame, 5)

    # Normal koşul
    return frame

