# Video evaluation

Video is evaluated as a bag of frames: representative frames are chosen from each video,
hashed with the existing CLIP image signal, and stored in HMA.

This folder currently holds the first two steps: choosing candidate frames
([frames.py](frames.py)) and filtering out low-information ones ([quality.py](quality.py)).

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
| `VIDEO_CANDIDATE_WINDOW_SECONDS` | How far either side of a slice's midpoint to also offer frames, so the quality filter can pick the sharpest. `0` offers the midpoint only | `0.25` |

Frames are decoded lazily, one at a time, so a long video is never held in memory.

The detector defaults were chosen by splicing real clips from different source videos
into longer videos with known cut positions (160 hard cuts, 6 fps). `adaptive` at 2.0 found
95% of the cuts at 95% precision; `adaptive` at its library default of 3.0 found 73%, and
`content` at 27.0 found 83% at 80% precision (and put a false cut in 16% of single-shot
clips, against 2% for `adaptive` at 2.0). This is hard cuts only: revisit the default with
footage that has dissolves or a different frame rate.

## Quality filtering

Each slice offers a small group of neighbouring frames (the midpoint and those within
`VIDEO_CANDIDATE_WINDOW_SECONDS`). A frame is junk if it is:

- **flat**: black, white or a solid colour (brightness barely varies),
- **bars**: vertical bars such as colour bars or a test pattern (adjacent rows are nearly identical), or
- **blurry**: too little edge sharpness (variance of the Laplacian).

Everything is measured on a 256-pixel-wide grayscale copy, so it is cheap and does not
depend on the video's resolution. The sharpest frame that is not junk is kept for each
slice. A slice with only junk is **dropped**, and a video in which every slice is junk
yields **no frames**: `QualityReport.summary()` says how many slices and frames were
dropped and why, and `no_usable_frames` tells the caller to tell the user.

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_QUALITY_ANALYSIS_WIDTH` | Width of the grayscale copy the checks run on | `256` |
| `VIDEO_FLAT_STD_MAX` | Brightness spread below this is flat | `5.0` |
| `VIDEO_BARS_ROW_DIFF_MAX` | Mean adjacent-row difference below this is bars | `0.7` |
| `VIDEO_BLUR_MIN` | Laplacian variance below this is blurry | `20.0` |

The defaults were set from 600 real frames (199 clips of the series dataset), synthetic
junk and blurred copies of the real frames. Real frames had a brightness spread of at
least 15 (black, white, solid and near-black frames: at most 0.5) and adjacent-row
differences of at least 0.98 (colour bars: 0.18 to 0.49), and a blur of 0.8% of the frame
width takes the median sharpness from about 880 to 8. On those clips the defaults flag
14 of 2,992 candidate frames, drop 1 of 600 slices and leave no video empty. They come
from one dataset, so check them on footage that looks different (night scenes, screen
recordings).

```python
from hma.video.frames import sample_frame_groups
from hma.video.quality import select_frames

selected, report = select_frames(sample_frame_groups("path/to/video.mp4"))
print(report.summary())
for item in selected:
    print(item.frame.index, round(item.frame.seconds, 2), round(item.quality.sharpness))
```

Try the sampling on a video (a 3 s clip with no cut gives three frames, at 0.5 s, 1.5 s and 2.5 s):

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
PYTHONPATH=. python -m pytest hma/unit_tests/test_video_frames.py hma/unit_tests/test_video_quality.py -v
```
