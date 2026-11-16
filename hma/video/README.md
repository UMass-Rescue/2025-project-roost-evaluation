# Video evaluation

Video is evaluated as a bag of frames: representative frames are chosen from each video,
hashed with the existing CLIP image signal, and stored in HMA.

This folder currently holds the first step, choosing candidate frames
([frames.py](frames.py)).

## Frame sampling

Scene cuts are found with [PySceneDetect](https://www.scenedetect.com/) (OpenCV backend).
Each scene is then cut into the fewest equal slices no longer than
`VIDEO_MAX_FRAME_GAP_SECONDS`, and the middle frame of every slice is taken. A scene
shorter than the gap gets one frame at its midpoint. With the detector set to `none`
the whole video is one scene, which is uniform sampling.

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_SCENE_DETECTOR` | `adaptive` (colour/brightness change compared with its neighbours, fewer false cuts on fast motion), `content` (change is large on its own) or `none` | `adaptive` |
| `VIDEO_SCENE_THRESHOLD` | Detector sensitivity, lower finds more cuts. Empty uses the detector's default: 2.0 for `adaptive`, 27.0 for `content` | empty |
| `VIDEO_MIN_SCENE_SECONDS` | Shortest allowed scene, in seconds so it behaves the same at any frame rate | `1.0` |
| `VIDEO_MAX_FRAME_GAP_SECONDS` | Longest gap between sampled frames inside a scene | `1.0` |

Frames are decoded lazily, one at a time, so a long video is never held in memory.

The detector defaults were chosen by splicing real clips from different source videos
into longer videos with known cut positions (160 hard cuts, 6 fps). `adaptive` at 2.0 found
95% of the cuts at 95% precision; `adaptive` at its library default of 3.0 found 73%, and
`content` at 27.0 found 83% at 80% precision (and put a false cut in 16% of single-shot
clips, against 2% for `adaptive` at 2.0). This is hard cuts only: revisit the default with
footage that has dissolves or a different frame rate.

Try it on a video (a 3 s clip with no cut gives three frames, at 0.5 s, 1.5 s and 2.5 s):

```bash
PYTHONPATH=. python - <<'EOF'
from hma.video.frames import sample_frames

for frame in sample_frames("path/to/video.mp4"):
    print(frame.index, round(frame.seconds, 2), frame.image.shape)
EOF
```

## Dependencies

- `opencv-python-headless` is pinned below 4.13 in [requirements.txt](../../requirements.txt):
  PySceneDetect 0.7.3 crashes on OpenCV 4.14 and 5.0 (the video position is NaN at the end
  of the stream), and 4.12 works.
- PySceneDetect itself is installed with `--no-deps` from
  [requirements-scenedetect.txt](../../requirements-scenedetect.txt), because it declares the
  non-headless `opencv-python`, which needs GUI libraries the evaluation image does not have.
  The Dockerfile and `make create-env` both do this. pip then prints a warning that
  `scenedetect requires opencv-python, which is not installed`, which is expected.

## Unit tests

The tests generate small synthetic videos on the fly, so they need no dataset:

```bash
pip install -r hma/requirements-dev.txt
PYTHONPATH=. python -m pytest hma/unit_tests/test_video_frames.py -v
```
