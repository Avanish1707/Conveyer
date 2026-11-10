"""Preview sack.pt detections on bag1.mp4 at confidence 0.35."""

from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("sack.pt")
VIDEO_PATH = Path("bag3.mp4")
CONFIDENCE = 0.50


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

model = YOLO(str(MODEL_PATH))
model.predict(
    source=str(VIDEO_PATH),
    conf=CONFIDENCE,
    show=True,
    save=True,
    project="inference_output",
    name="bag3",
    exist_ok=True,
)

print("Finished. Annotated video saved in inference_output/bag1.")
