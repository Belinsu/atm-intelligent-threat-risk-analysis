import cv2
import numpy as np

def preprocess_face(face_img):
    if face_img is None:
        return None

    # çok küçük yüzleri alma
    if face_img.shape[0] < 80 or face_img.shape[1] < 80:
        return None

    # 1. Gamma düzeltme
    gamma = 1.4
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in range(256)]
    ).astype("uint8")
    face_img = cv2.LUT(face_img, table)

    # 2. CLAHE (ışık kanalı)
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    face_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # 3. hafif netleştirme
    blur = cv2.GaussianBlur(face_img, (0, 0), sigmaX=1.0)
    face_img = cv2.addWeighted(face_img, 1.5, blur, -0.5, 0)

    return face_img
