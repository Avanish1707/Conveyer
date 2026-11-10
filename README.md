# Conveyor bag counting

`bag_counter.py` lets you draw one polygon around the conveyor. It detects people over the **whole camera frame** and detects/tracks sugar sacks, counting each sack once when it is inside the conveyor polygon.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run with the three videos

```powershell
python bag_counter.py --save-video
```

That command uses Grounding DINO with the built-in `person` and `white woven
sack` labels. You do not need to provide any detection text: only draw the
conveyor polygon when the ROI window opens. The first run downloads about 1 GB
of model weights and is considerably slower without an NVIDIA GPU.

With no video arguments, the script automatically uses every `bag*.mp4` file in the current folder. The annotated preview is shown by default; use `--no-show` for headless processing.

To replace the saved conveyor polygon, run:

```powershell
python bag_counter.py --redraw-roi
```

The overlay reports unique people in the full frame, plus the people currently seen in/out of the ROI. Sugar sacks are counted only once their tracked reference point is inside the ROI. By default a sack must be tracked for three frames before it is counted, avoiding most one-frame false positives; adjust this with `--min-bag-track-frames`.

## Quick sugar-bag test

The normal `yolo11n.pt` model does not contain a sugar-bag class. For a no-training experiment, use YOLO-World:

```powershell
python bag_counter.py --redraw-roi --save-video
```

YOLO-World downloads its weights the first time. Prompt detection may miss or confuse similar sacks. For dependable counting, label several hundred frames from this conveyor with bounding boxes and train a custom model, then run:

```powershell
python bag_counter.py --model .\weights\best.pt --bag-classes bag --redraw-roi
```

Person IDs use the persistent BoT-SORT tracker, so an ID remains the same while the person stays trackable. If a person is fully hidden for too long or leaves and later re-enters, a new ID can be assigned; permanent identity across separate videos requires a ReID model and is not guaranteed by ordinary tracking.

On the first run, click at least three points around the conveyor, press `Enter`, and the polygon is saved to `roi.json`. The same polygon is reused for the next runs. Delete `roi.json` to draw it again.

The script prints a bag count for each video and a total. It also reports unique tracked people that entered the ROI. Press `Esc` during processing to stop.

## Important model note

The default `yolo11n.pt` model is a general COCO model. It recognizes `backpack`, `handbag`, and `suitcase`, but generic industrial bags may not belong to those classes. For accurate conveyor results, train a YOLO model with a `bag` class and run:

```powershell
python bag_counter.py .\bag1.mp4 .\bag2.mp4 .\bag3.mp4 --model .\weights\best.pt --bag-classes bag --show --save-video
```

Human detection uses the `person` class. The count is based on unique tracker IDs entering the polygon, so a bag is counted once even while it remains in the ROI.
