"""Helpers shared by the video unit tests: synthetic frames and videos built on the fly."""

import cv2
import numpy as np

H, W = 360, 640


def detailed(seed=0, block=40, noise=20):
    """A sharp frame with large-scale structure and fine detail, like a busy real scene.

    The same seed always gives the same frame, and different seeds give clearly different ones.
    Its sharpness falls with blur much as a real frame's does (about 1,500, then 55 at 0.4% blur
    and 10 at 0.8%)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H // block + 1, W // block + 1, 3), dtype=np.uint8)
    image = cv2.resize(base, (W + block, H + block), interpolation=cv2.INTER_NEAREST)[:H, :W]
    return np.clip(image.astype(np.float32) + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)


def write_video(path, frames, fps=10):
    """Write frames (BGR arrays of one size) to a video and return its path as a string.

    MJPG in an .avi container is used because OpenCV can write it with both its macOS
    (AVFoundation) and Linux (FFmpeg) backends; mp4v/.mp4 is not available on macOS."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened(), "OpenCV cannot write MJPG/avi video in this environment"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(path)


def scenes_video(path, scenes, fps=10):
    """A video of scenes, each a (seed, frame_count) pair: the frame `detailed(seed)` held for
    frame_count frames, so a scene is perfectly static and two scenes differ completely."""
    frames = []
    for seed, count in scenes:
        frames.extend([detailed(seed)] * count)
    return write_video(path, frames, fps)
