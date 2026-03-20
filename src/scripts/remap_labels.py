import os

# eski → yeni sınıf id'leri
OLD_NAMES = {
    0: "balaclava",
    1: "medicine mask",
    2: "scarf",
    3: "hat",
    4: "hand",
    5: "nothing"
}

NEW_NAMES = {
    "balaclava": 0,
    "medicine mask": 1,
    "scarf": 2,
    "head_cover": 3,
    "hand": 4,
    "nothing": 5
}

LABEL_DIR = "data/face_concealment_v3/train/labels"

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, file)
    new_lines = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])

            old_name = OLD_NAMES.get(cls)

            if old_name == "hat":
                new_cls = NEW_NAMES["head_cover"]
            else:
                new_cls = NEW_NAMES.get(old_name)

            new_lines.append(
                " ".join([str(new_cls)] + parts[1:])
            )

    with open(path, "w") as f:
        f.write("\n".join(new_lines))

