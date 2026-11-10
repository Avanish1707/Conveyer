"""Detect people in the full frame and count sugar sacks in a conveyor ROI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


DEFAULT_VIDEO = Path("bag1.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Input MP4/video files; if omitted, all bag*.mp4 files are used",
    )
    parser.add_argument("--bag-model", default="sack.pt", help="YOLO weights for sugar sack detection")
    parser.add_argument("--person-model", default="yolo11n.pt", help="YOLO weights for person detection")
    parser.add_argument("--tracker", default="botsort.yaml", help="Ultralytics tracker configuration")
    parser.add_argument("--bag-classes", default="sugar_sack", help="Model class names treated as sugar sacks")
    parser.add_argument("--roi", type=Path, default=Path("roi.json"), help="Saved polygon file")
    parser.add_argument("--redraw-roi", action="store_true", help="Draw a new polygon instead of using the saved ROI")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--bag-confidence", type=float, default=0.90, help="Confidence threshold for sack detection")
    parser.add_argument("--motion-threshold", type=int, default=18, help="Pixel-change threshold for conveyor motion")
    parser.add_argument("--motion-pixel-ratio", type=float, default=0.02, help="Changed-pixel ratio required for a running conveyor")
    parser.add_argument("--motion-still-frames", type=int, default=15, help="Still frames required before marking the conveyor stopped")
    parser.add_argument(
        "--min-bag-track-frames",
        type=int,
        default=3,
        help="Require a sack to be tracked for this many frames before counting it",
    )
    parser.add_argument("--show", dest="show", action="store_true", default=True, help="Display annotated video (default)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not display the annotated video")
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True, help="Save annotated MP4 files")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="Do not save annotated MP4 files")
    return parser.parse_args()


def select_polygon(video_path: Path, roi_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame from {video_path}")

    points: list[tuple[int, int]] = []
    window = "Draw ROI: click points, ENTER to accept, R to reset, ESC to cancel"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        if points:
            polygon = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [polygon], len(points) > 2, (0, 255, 255), 2)
            for point in points:
                cv2.circle(canvas, point, 5, (0, 0, 255), -1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            break
        if key in (ord("r"), ord("R")):
            points.clear()
        if key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("ROI selection cancelled")
    cv2.destroyAllWindows()

    height, width = frame.shape[:2]
    roi_path.write_text(json.dumps({"points": points, "frame_size": [width, height]}, indent=2), encoding="utf-8")
    return np.array(points, dtype=np.int32), (width, height)


def load_or_select_roi(video_path: Path, roi_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    if roi_path.exists():
        data = json.loads(roi_path.read_text(encoding="utf-8"))
        points = np.array(data["points"], dtype=np.int32)
        if len(points) >= 3:
            frame_size = tuple(data.get("frame_size", [0, 0]))
            if frame_size[0] > 0 and frame_size[1] > 0:
                return points, frame_size
    return select_polygon(video_path, roi_path)


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    return names[class_id]


def available_class_names(names: dict[int, str] | list[str]) -> set[str]:
    return {name.lower() for name in names.values()} if isinstance(names, dict) else {name.lower() for name in names}


class GroundingDinoTracker:
    """Small IoU tracker that makes Grounding DINO compatible with this POC."""

    def __init__(self, model_id: str, labels: list[str]) -> None:
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(
                "Grounding DINO needs Transformers. Run: python -m pip install -r requirements.txt"
            ) from error
        self.labels = labels
        self.names = {index: label for index, label in enumerate(labels)}
        self.detector = pipeline("zero-shot-object-detection", model=model_id, device=-1)
        self.tracks: dict[int, tuple[int, tuple[int, int, int, int], int]] = {}
        self.next_id = 1

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - intersection
        return intersection / union if union else 0.0

    def track(self, frame: np.ndarray, *, conf: float, **_: Any) -> list[Any]:
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        predictions = self.detector(image, candidate_labels=self.labels, threshold=conf)
        detections: list[tuple[int, tuple[int, int, int, int], float]] = []
        for prediction in predictions:
            label = prediction["label"].lower()
            if label not in self.labels:
                continue
            box = prediction["box"]
            bbox = (int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"]))
            detections.append((self.labels.index(label), bbox, float(prediction["score"])))

        matched_tracks: set[int] = set()
        ids: list[int] = []
        for class_id, bbox, _score in detections:
            best_id, best_iou = None, 0.0
            for track_id, (tracked_class, tracked_box, _age) in self.tracks.items():
                overlap = self._iou(bbox, tracked_box)
                if track_id not in matched_tracks and tracked_class == class_id and overlap > best_iou:
                    best_id, best_iou = track_id, overlap
            if best_id is None or best_iou < 0.30:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = (class_id, bbox, 0)
            matched_tracks.add(best_id)
            ids.append(best_id)
        for track_id, (class_id, bbox, age) in list(self.tracks.items()):
            if track_id not in matched_tracks:
                self.tracks[track_id] = (class_id, bbox, age + 1)
                if age >= 30:
                    del self.tracks[track_id]

        class Boxes:
            def __init__(self) -> None:
                self.xyxy = torch.tensor([item[1] for item in detections], dtype=torch.float32).reshape(-1, 4)
                self.id = torch.tensor(ids, dtype=torch.int64)
                self.cls = torch.tensor([item[0] for item in detections], dtype=torch.int64)
                self.conf = torch.tensor([item[2] for item in detections], dtype=torch.float32)

            def __len__(self) -> int:
                return len(detections)

        class Result:
            boxes = Boxes()

        return [Result()]


def assign_person_id(
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    person_tracks: dict[int, dict[str, Any]],
    next_person_id: int,
    used_person_ids: set[int],
) -> tuple[int, int]:
    """Match a detection to a nearby logical person when the tracker ID changes."""
    best_id = None
    best_distance = float("inf")
    x1, y1, x2, y2 = bbox
    detection_width = max(x2 - x1, 1)
    detection_height = max(y2 - y1, 1)
    for person_id, track in person_tracks.items():
        if person_id in used_person_ids:
            continue
        previous_center = track["center"]
        previous_bbox = track["bbox"]
        distance = float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]))
        px1, py1, px2, py2 = previous_bbox
        overlap_width = max(0, min(x2, px2) - max(x1, px1))
        overlap_height = max(0, min(y2, py2) - max(y1, py1))
        overlap = (overlap_width * overlap_height) / (detection_width * detection_height)
        allowed_distance = max(100.0, detection_width * 0.75, detection_height * 0.5)
        if (distance <= allowed_distance or overlap >= 0.15) and distance < best_distance:
            best_id = person_id
            best_distance = distance

    if best_id is None:
        best_id = next_person_id
        next_person_id += 1
    person_tracks[best_id] = {"center": center, "bbox": bbox, "missing": 0}
    return best_id, next_person_id


def assign_bag_id(
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    bag_tracks: dict[int, dict[str, Any]],
    next_bag_id: int,
    used_bag_ids: set[int],
) -> tuple[int, int]:
    """Keep a sack's logical ID when its detector tracker ID changes."""
    best_id = None
    best_distance = float("inf")
    x1, y1, x2, y2 = bbox
    detection_width = max(x2 - x1, 1)
    detection_height = max(y2 - y1, 1)
    for bag_id, track in bag_tracks.items():
        if bag_id in used_bag_ids:
            continue
        previous_center = track["center"]
        previous_bbox = track["bbox"]
        distance = float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]))
        px1, py1, px2, py2 = previous_bbox
        overlap_width = max(0, min(x2, px2) - max(x1, px1))
        overlap_height = max(0, min(y2, py2) - max(y1, py1))
        overlap = (overlap_width * overlap_height) / (detection_width * detection_height)
        allowed_distance = max(55.0, detection_width * 0.45, detection_height * 0.30)
        if (distance <= allowed_distance or overlap >= 0.15) and distance < best_distance:
            best_id = bag_id
            best_distance = distance

    if best_id is None:
        best_id = next_bag_id
        next_bag_id += 1
    return best_id, next_bag_id


def process_video(
    video_path: Path,
    bag_model: Any,
    person_model: Any,
    roi: np.ndarray,
    roi_frame_size: tuple[int, int],
    bag_classes: set[str],
    args: argparse.Namespace,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = width / roi_frame_size[0]
    scale_y = height / roi_frame_size[1]
    scaled_roi = np.round(roi * [scale_x, scale_y]).astype(np.int32)
    conveyor_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(conveyor_mask, [scaled_roi], 255)
    conveyor_pixel_count = max(cv2.countNonZero(conveyor_mask), 1)
    previous_gray: np.ndarray | None = None
    conveyor_running = True
    stop_start_frame: int | None = None
    still_frames = 0
    moving_frames = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.save_video:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(args.output_dir / f"{video_path.stem}_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    bag_tracks: dict[int, dict[str, Any]] = {}
    counted_bag_ids: set[int] = set()
    bag_tracker_ids: dict[int, int] = {}
    human_inside_ids: set[int] = set()
    human_outside_ids: set[int] = set()
    person_tracks: dict[int, dict[str, Any]] = {}
    person_tracker_ids: dict[int, int] = {}
    sacks_in_roi: set[int] = set()
    next_person_id = 1
    next_bag_id = 1
    bag_count = 0
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            difference = cv2.absdiff(previous_gray, gray)
            changed_pixels = np.where(difference >= args.motion_threshold, 255, 0).astype(np.uint8)
            motion_pixels = cv2.countNonZero(cv2.bitwise_and(changed_pixels, conveyor_mask))
            motion_detected = motion_pixels / conveyor_pixel_count >= args.motion_pixel_ratio
            if motion_detected:
                moving_frames += 1
                still_frames = 0
                if not conveyor_running and moving_frames >= 2:
                    conveyor_running = True
                    stop_start_frame = None
            else:
                still_frames += 1
                moving_frames = 0
                if conveyor_running and still_frames >= args.motion_still_frames:
                    conveyor_running = False
                    stop_start_frame = frame_number - still_frames + 1
        previous_gray = gray

        bag_result = bag_model.track(
            frame,
            persist=frame_number > 0,
            tracker=args.tracker,
            conf=args.bag_confidence,
            verbose=False,
        )[0]
        person_result = person_model.track(
            frame,
            persist=frame_number > 0,
            tracker=args.tracker,
            conf=args.confidence,
            verbose=False,
            classes=[0],
        )[0]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [scaled_roi], (0, 120, 180))
        frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
        cv2.polylines(frame, [scaled_roi], True, (0, 255, 255), 2)

        boxes = person_result.boxes
        for track in person_tracks.values():
            track["missing"] += 1
        if boxes is not None and boxes.xyxy is not None:
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
            matched_person_ids: set[int] = set()
            for box, track_id, cls_tensor, confidence in zip(
                boxes.xyxy.int().cpu().tolist(),
                ids,
                boxes.cls.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
            ):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                inside = cv2.pointPolygonTest(scaled_roi, center, False) >= 0
                person_id = person_tracker_ids.get(track_id)
                if person_id is None:
                    person_id, next_person_id = assign_person_id(
                        (x1, y1, x2, y2),
                        center,
                        person_tracks,
                        next_person_id,
                        matched_person_ids,
                    )
                    if track_id >= 0:
                        person_tracker_ids[track_id] = person_id
                if person_id not in person_tracks:
                    person_tracks[person_id] = {
                        "center": center,
                        "bbox": (x1, y1, x2, y2),
                        "missing": 0,
                    }
                else:
                    person_tracks[person_id].update(
                        {"center": center, "bbox": (x1, y1, x2, y2)}
                    )
                matched_person_ids.add(person_id)
                if inside:
                    human_inside_ids.add(person_id)
                else:
                    human_outside_ids.add(person_id)
                person_tracks[person_id]["missing"] = 0
                color = (0, 255, 0) if inside else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"person ID:{person_id} conf:{confidence:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                cv2.circle(frame, center, 4, color, -1)

        bag_boxes = bag_result.boxes
        sacks_in_roi.clear()
        matched_bag_ids: set[int] = set()
        if bag_boxes is not None and bag_boxes.xyxy is not None:
            bag_ids = bag_boxes.id.int().cpu().tolist() if bag_boxes.id is not None else [-1] * len(bag_boxes)
            for box, track_id, cls_tensor, confidence in zip(
                bag_boxes.xyxy.int().cpu().tolist(),
                bag_ids,
                bag_boxes.cls.int().cpu().tolist(),
                bag_boxes.conf.cpu().tolist(),
            ):
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = box
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                roi_reference = ((x1 + x2) // 2, y2)
                inside = cv2.pointPolygonTest(scaled_roi, roi_reference, False) >= 0
                name = class_name(bag_model.names, cls_tensor)
                if name.lower() not in bag_classes:
                    continue
                bag_id = bag_tracker_ids.get(track_id)
                if bag_id is None:
                    bag_id, next_bag_id = assign_bag_id(
                        (x1, y1, x2, y2),
                        center,
                        bag_tracks,
                        next_bag_id,
                        matched_bag_ids,
                    )
                    if track_id >= 0:
                        bag_tracker_ids[track_id] = bag_id
                if bag_id in counted_bag_ids:
                    continue
                matched_bag_ids.add(bag_id)
                state = bag_tracks.get(bag_id)
                if state is None and not inside:
                    continue
                if state is None:
                    state = {
                        "frames": 0,
                        "entered_roi": False,
                        "center": center,
                        "bbox": (x1, y1, x2, y2),
                        "outside_frames": 0,
                    }
                    bag_tracks[bag_id] = state
                else:
                    state.update({"center": center, "bbox": (x1, y1, x2, y2)})
                state["frames"] += 1
                if inside:
                    sacks_in_roi.add(bag_id)
                    state["entered_roi"] = True
                    state["outside_frames"] = 0
                elif state["entered_roi"]:
                    state["outside_frames"] += 1
                if (
                    state["entered_roi"]
                    and state["outside_frames"] >= args.min_bag_track_frames
                ):
                    bag_count += 1
                    counted_bag_ids.add(bag_id)
                    del bag_tracks[bag_id]
                if inside:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} ID:{bag_id} conf:{confidence:.2f}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    cv2.circle(frame, center, 4, color, -1)

        cv2.putText(
            frame,
            f"Sacks in ROI: {len(sacks_in_roi)}  Total sacks: {bag_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        total_humans = len(human_inside_ids | human_outside_ids)
        cv2.putText(
            frame,
            f"Total humans: {total_humans}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
        current_stop_frames = frame_number - stop_start_frame + 1 if stop_start_frame is not None else 0
        current_stop_seconds = current_stop_frames / fps
        conveyor_color = (0, 255, 0) if conveyor_running else (0, 0, 255)
        cv2.putText(
            frame,
            f"Conveyor: {'RUNNING' if conveyor_running else 'STOPPED'}  Stop: {current_stop_seconds:.1f}s",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            conveyor_color,
            2,
        )
        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("Bag and human detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_number += 1

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(
        f"{video_path.name}: sugar_sacks_in_roi={bag_count}, "
        f"unique_people_full_frame={len(human_inside_ids | human_outside_ids)}, "
        f"unique_people_in_roi={len(human_inside_ids)}, "
        f"unique_people_outside_roi={len(human_outside_ids)}"
    )
    return bag_count


def main() -> None:
    args = parse_args()
    if not args.videos:
        args.videos = [DEFAULT_VIDEO]
    for video in args.videos:
        if not video.exists():
            raise FileNotFoundError(video)
    if args.redraw_roi:
        roi, roi_frame_size = select_polygon(args.videos[0], args.roi)
    else:
        roi, roi_frame_size = load_or_select_roi(args.videos[0], args.roi)
    bag_classes = {name.strip().lower() for name in args.bag_classes.split(",") if name.strip()}
    bag_model_path = Path(args.bag_model)
    person_model_path = Path(args.person_model)
    for model_path in (bag_model_path, person_model_path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    bag_model = YOLO(str(bag_model_path))
    person_model = YOLO(str(person_model_path))
    model_classes = available_class_names(bag_model.names)
    configured_bag_classes = bag_classes & model_classes
    if not configured_bag_classes:
        print(
            "WARNING: no configured bag class exists in this model. "
            f"Configured: {sorted(bag_classes)}; model classes include: {sorted(model_classes)}"
        )
        print("Use a custom model trained with a bag/sack class and pass --bag-classes bag.")
    total = sum(
        process_video(video, bag_model, person_model, roi, roi_frame_size, bag_classes, args)
        for video in args.videos
    )
    print(f"TOTAL bags across {len(args.videos)} videos: {total}")


if __name__ == "__main__":
    main()
