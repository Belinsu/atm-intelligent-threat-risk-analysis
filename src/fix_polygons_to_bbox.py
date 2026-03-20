from pathlib import Path

labels_dir = Path("data/face_concealment_v3/train/labels")

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

fixed = 0
skipped = 0
bad = 0

for p in labels_dir.glob("*.txt"):
    lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not lines:
        continue

    new_lines = []
    changed = False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        cls = parts[0]
        nums = parts[1:]

        # Already bbox (x y w h)
        if len(nums) == 4:
            new_lines.append(f"{cls} " + " ".join(nums))
            continue

        # Polygon/seg format: expect even count (x1 y1 x2 y2 ...)
        if len(nums) < 6 or len(nums) % 2 != 0:
            bad += 1
            # skip this annotation line to avoid corrupting training
            # (or you can raise)
            continue

        coords = list(map(float, nums))
        xs = coords[0::2]
        ys = coords[1::2]

        x_min = clamp01(min(xs))
        x_max = clamp01(max(xs))
        y_min = clamp01(min(ys))
        y_max = clamp01(max(ys))

        # Convert to YOLO bbox (cx cy w h)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        w  = max(1e-6, x_max - x_min)
        h  = max(1e-6, y_max - y_min)

        new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        changed = True

    if new_lines:
        if changed:
            p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            fixed += 1
        else:
            skipped += 1
    else:
        # If all lines were bad, keep file but empty -> remove file to prevent cache issues
        p.write_text("", encoding="utf-8")
        bad += 1

print("DONE")
print("fixed_files:", fixed)
print("unchanged_files:", skipped)
print("bad_lines_or_empty_files:", bad)
