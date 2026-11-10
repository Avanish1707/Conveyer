"""Run the sugar-sack YOLO model on one or more videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Input video files; if omitted, all MP4 files in this folder are used",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("sack.pt"),
        help="YOLO model weights",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inference_output"))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size; increase for small sacks")
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display the annotated video while processing (default)",
    )
    parser.add_argument("--no-show", dest="show", action="store_false", help="Do not display the preview")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated videos")
    return parser.parse_args()


def infer_video(video_path: Path, model: YOLO, args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(
            str(args.output_dir / f"{video_path.stem}_inferred.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=args.confidence,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )[0]
        annotated_frame = result.plot()

        if writer is not None:
            writer.write(annotated_frame)
        if args.show:
            cv2.imshow("Sugar sack inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_number += 1

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    destination = "not saved" if args.no_save else str(args.output_dir / f"{video_path.stem}_inferred.mp4")
    print(f"{video_path.name}: processed {frame_number} frames -> {destination}")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    videos = args.videos or sorted(Path.cwd().glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("No MP4 videos found. Pass a video path on the command line.")
    for video in videos:
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

    model = YOLO(str(args.model))
    print(f"Loaded model: {args.model}")
    print(f"Model classes: {model.names}")
    for video in videos:
        infer_video(video, model, args)


if __name__ == "__main__":
    main()