import yaml

with open("main.py", "r") as f:
    content = f.read()

if "WeaponDetector" in content:
    print("⚠️ Weapon zaten entegre görünüyor.")
    exit()

inject_imports = """
import yaml
from utils.weapon_detector import WeaponDetector
from utils.risk_engine import weapon_risk_score
"""

content = inject_imports + content

inject_init = """
with open("weapon_config.yaml") as f:
    weapon_cfg = yaml.safe_load(f)

weapon_detector = WeaponDetector(weapon_cfg)
"""

content = content.replace(
    "cap = cv2.VideoCapture",
    inject_init + "\ncap = cv2.VideoCapture"
)

inject_loop = """
    weapon_detections = weapon_detector.detect(frame)
    weapon_score = weapon_risk_score(weapon_detections)

    for d in weapon_detections:
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["label"].upper()} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    if weapon_score >= 5:
        cv2.putText(frame, "HIGH RISK",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,0,255), 3)
"""

content = content.replace(
    "cv2.imshow",
    inject_loop + "\n    cv2.imshow"
)

with open("main.py", "w") as f:
    f.write(content)

print("✅ Weapon detection main.py içine otomatik eklendi")
