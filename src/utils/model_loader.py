import os
import subprocess

MODEL_PATH = "models/face_concealment_hat_best.pt"
MODEL_URL = "https://drive.google.com/uc?id=1BIumVDWxb45ahjACsX5Al8yEnwMWUBvm"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Model indiriliyor...")
        subprocess.run(["pip", "install", "-q", "gdown"])
        subprocess.run(["gdown", MODEL_URL, "-O", MODEL_PATH])
    else:
        print("✅ Model zaten mevcut")

if __name__ == "__main__":
    ensure_model()

