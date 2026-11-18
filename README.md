# ATM Intelligent Threat & Risk Analysis System

A real-time computer vision prototype that watches an ATM camera feed, detects suspicious visual signals around each person, and turns them into a per-person risk score from 0 to 100%.

## What it does

For every frame from the camera, the system:

1. **Detects faces and hands** with a YOLOv8 face-hiding model. If no face is visible at all, it flags possible concealment.
2. **Classifies face concealment** on each face crop with a second YOLOv8 model: balaclava, medical mask, scarf, hat, hood or cap.
3. **Detects weapons** (gun, knife) with a separate YOLOv8 detector.
4. **Stabilizes labels** over the last 7 frames by majority vote, so a single noisy frame does not change the result.
5. **Scores risk per person** by combining the visual signals with behavioral ones: how long the person has been in view and how many people are around the ATM.
6. **Draws the result** on the live feed: a box and risk percentage for each person, colored green, yellow or red.

## Risk scoring

| Signal | Effect on risk |
| --- | --- |
| Weapon detected (gun or knife) | Risk set to 100% |
| Balaclava | At least 70% |
| Hood or cap combined with a mask or scarf | At least 55% |
| Medical mask or scarf | +25 |
| Hat, hood or cap | +25 |
| Hand detected | +15 |
| In view for more than 30 seconds | +5 for every further 10 seconds |
| More than 2 people in view | +10 for each extra person |

The score is capped at 100%. On screen, **50% and above** is shown in yellow and **80% and above** in red.

## Model training

- Custom **YOLOv8n** face-concealment model trained on a dataset of **3,600+ images** curated and labeled with Roboflow.
- Trained on GPU in Google Colab and Kaggle.
- Reached **mAP@50 ≈ 0.80**.
- Validated on real-world videos and images.

Dataset sources and export notes are in [`configs/`](configs) and [`docs/`](docs).

## Tech stack

Python · YOLOv8 (Ultralytics) · OpenCV · Roboflow · Google Colab · Kaggle

## Repository structure

| Path | Contents |
| --- | --- |
| [`src/main.py`](src/main.py) | Real-time pipeline: camera input, detection, label stabilization, risk scoring and display |
| [`src/utils/`](src/utils) | Weapon detector, weapon risk rule, image preprocessing and model download helper |
| [`src/test_image.py`](src/test_image.py), [`src/test_images_batch.py`](src/test_images_batch.py) | Run the models on single images or a batch of images |
| [`src/test_video.py`](src/test_video.py), [`src/test_video_risk.py`](src/test_video_risk.py) | Run detection and risk scoring on video files |
| [`src/scripts/`](src/scripts), [`src/fix_polygons_to_bbox.py`](src/fix_polygons_to_bbox.py) | Dataset preparation: label remapping and polygon-to-box conversion |
| [`configs/`](configs) | Dataset YAML files for training |
| [`docs/`](docs) | Dataset and Roboflow export notes |
| [`datasets/`](datasets) | Note on the omitted dataset files |

## Running it

Trained weights, images, videos and dataset splits are not included in this repository because of their size.

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the trained weights in `src/models/`:

   ```text
   src/models/face_hiding_best.pt
   src/models/face_concealment_hat_best.pt
   src/models/weapon/weapon_best.pt
   ```

   The face-concealment model can be downloaded with `python utils/model_loader.py`, run from inside `src/`.

3. Start the real-time pipeline from inside `src/` (it uses the default webcam; press `Q` to quit):

   ```bash
   cd src
   python main.py
   ```

## Possible next steps

- Track people across frames with a proper tracker instead of box positions
- Send alerts for high-risk events to a monitoring dashboard
- Tune the risk weights on labeled incident footage
