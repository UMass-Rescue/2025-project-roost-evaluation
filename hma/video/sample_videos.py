"""Deterministic sample videos for trying the video evaluation without the real dataset.

Each scene is a coloured picture with a few shapes and a word, which CLIP tells apart from every
other scene. A series is four scenes, and its three clips show them two at a time (1 and 2, 2 and 3,
3 and 4), so neighbouring clips share a scene and clips of different series share none. The same call
always writes the same files.

    python -m hma.video.sample_videos path/to/folder
"""

import sys
from pathlib import Path

import cv2
import numpy as np

HEIGHT, WIDTH = 360, 640
FPS = 10
SCENE_SECONDS = 1.0
SCENES_PER_SERIES = 4
CLIPS_PER_SERIES = SCENES_PER_SERIES - 1

WORDS = [
    "apple", "boat", "castle", "dragon", "engine", "forest", "guitar", "harbor",
    "island", "jungle", "kitchen", "lantern", "mountain", "network", "ocean", "palace",
]


def scene_image(scene: int) -> np.ndarray:
    """The BGR picture of one scene; the same number always gives the same picture."""
    rng = np.random.default_rng(scene)
    background = cv2.cvtColor(np.uint8([[[(scene * 37) % 180, 160, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    image[:] = background
    darker = np.linspace(0, 80, WIDTH, dtype=np.uint8)[None, :, None]
    image = np.clip(image.astype(np.int16) - darker, 0, 255).astype(np.uint8)
    for shape in range(3):
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(80, WIDTH - 80)), int(rng.integers(60, HEIGHT - 60))
        if shape % 2:
            cv2.circle(image, (x, y), int(rng.integers(30, 70)), colour, -1)
        else:
            cv2.rectangle(image, (x - 60, y - 40), (x + 60, y + 40), colour, -1)
    word = WORDS[scene % len(WORDS)]
    cv2.putText(image, word, (40, HEIGHT // 2), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 6)
    cv2.putText(image, word, (40, HEIGHT // 2), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 2)
    texture = rng.normal(0, 6, image.shape)  # fine detail, as in a real frame
    return np.clip(image + texture, 0, 255).astype(np.uint8)


def write_clip(path: Path, scenes: list[int]) -> None:
    """Write a clip that shows each scene for SCENE_SECONDS. MJPG in an .avi container is used
    because OpenCV can write it on macOS and on Linux."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot write MJPG/avi video in this environment")
    for scene in scenes:
        image = scene_image(scene)
        for _ in range(round(FPS * SCENE_SECONDS)):
            writer.write(image)
    writer.release()


def make_sample_videos(folder: Path, series: int = 4) -> list[str]:
    """Write `series` series folders of three clips each into folder and return the clip paths."""
    paths = []
    for s in range(series):
        series_dir = Path(folder) / f"series_{s + 1:04d}"
        series_dir.mkdir(parents=True, exist_ok=True)
        first_scene = s * SCENES_PER_SERIES
        for clip in range(CLIPS_PER_SERIES):
            path = series_dir / f"clip_{clip + 1:03d}.avi"
            write_clip(path, [first_scene + clip, first_scene + clip + 1])
            paths.append(str(path))
    return paths


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    written = make_sample_videos(Path(sys.argv[1]))
    print(f"Wrote {len(written)} clips in {len(written) // CLIPS_PER_SERIES} series to {sys.argv[1]}")
