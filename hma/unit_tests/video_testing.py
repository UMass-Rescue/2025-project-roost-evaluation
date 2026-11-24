"""Helpers shared by the video unit tests: synthetic frames and videos built on the fly."""

import cv2
import numpy as np

from hma.video.frame_index import FrameRef, decode_signal, encode_signal
from hma.video.frames import SamplingConfig
from hma.video.pipeline import PipelineConfig

H, W = 360, 640

# bank items for tests of the matching: 1-2 are frames of video "a", 3 of "b", 4 of "c", 5 of "d"
FRAMES = {
    "1": FrameRef("a", 0, 0.5),
    "2": FrameRef("a", 6, 1.5),
    "3": FrameRef("b", 0, 0.5),
    "4": FrameRef("c", 0, 0.5),
    "5": FrameRef("d", 0, 0.5),
}

# one frame per second, no scene detection, no neighbouring candidates: easy to count
CONFIG = PipelineConfig(sampling=SamplingConfig(scene_detector="none", max_frame_gap_seconds=1.0, candidate_window_seconds=0))


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


class FakeHMA:
    """A stand-in for the HMA service: hashes a frame as its normalised 8x8 grayscale picture and keeps the bank in memory."""

    def __init__(self):
        self.bank = {}  # content id -> signal
        self.hash_calls = []
        self.fail_adds = False

    def hash_image_bytes(self, images, signal_type):
        self.hash_calls.append((len(images), signal_type))
        signals = []
        for image in images:
            gray = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
            small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            small -= small.mean()
            signals.append(encode_signal(small / np.linalg.norm(small)))
        return signals

    def add_signal_to_bank(self, bank_name, signal_type, signal):
        if self.fail_adds:
            return None
        content_id = str(len(self.bank) + 1)
        self.bank[content_id] = signal
        return content_id

    def match_with_signal_threshold(self, signal, threshold, signal_type):
        query = decode_signal(signal)
        matches = []
        for content_id, stored in self.bank.items():
            distance = 1 - float(query @ decode_signal(stored))
            if distance <= threshold:
                matches.append({"bank_content_id": int(content_id), "distance": str(distance)})
        return {"status": "success", "matches": matches}
