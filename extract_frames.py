"""Extract 125 chronological frames from bag1.mp4, bag2.mp4, and bag3.mp4."""

from pathlib import Path

import cv2


VIDEO_NAMES = ("bag1.mp4", "bag2.mp4", "bag3.mp4")
FRAMES_PER_VIDEO = 125
OUTPUT_ROOT = Path("extracted_frames")


def extract_frames(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame count: {video_path}")

    frame_count = min(FRAMES_PER_VIDEO, total_frames)
    selected_frames = {
        round(index * (total_frames - 1) / max(frame_count - 1, 1))
        for index in range(frame_count)
    }
    output_dir = OUTPUT_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number in selected_frames:
            output_path = output_dir / f"{saved_count + 1:04d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                capture.release()
                raise RuntimeError(f"Could not save frame: {output_path}")
            saved_count += 1
        frame_number += 1

    capture.release()
    print(f"{video_path.name}: saved {saved_count} frames to {output_dir}")
    return saved_count


def main() -> None:
    total_saved = 0
    for video_name in VIDEO_NAMES:
        video_path = Path(video_name)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        total_saved += extract_frames(video_path)
    print(f"Finished: saved {total_saved} frames in chronological video order.")


if __name__ == "__main__":
    main()
