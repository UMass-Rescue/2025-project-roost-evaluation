# Frame selection

How a video becomes the frames that are hashed. The steps run in order inside
[pipeline.py](../pipeline.py): choosing candidate frames ([frames.py](../frames.py)), filtering out
low-information ones ([quality.py](../quality.py)), and removing repeats ([dedupe.py](../dedupe.py)).
Where "the series dataset" is mentioned it is the dataset described in the [README](../README.md).
Storing the frames in HMA and scoring a query are described in
[EVALUATION_DESCRIPTION.md](EVALUATION_DESCRIPTION.md).

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

Frames are decoded lazily, one slice at a time.

The detector defaults were chosen on clips of the series dataset spliced together, so that
the cut positions were known: `adaptive` at 2.0 found 95% of 160 hard cuts at 95% precision,
against 83% at 80% for `content`. This covers hard cuts at 6 fps only, so revisit it for dissolves or other frame rates.

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

The thresholds are set so that black, white and colour-bar frames and heavily blurred frames
are caught while real frames pass. On 199 clips of the series dataset (2,992 candidate
frames) only 14 frames were flagged as junk, and every video kept frames. That is one dataset, so check the
thresholds on footage that looks different, such as night scenes or screen recordings.

```python
from hma.video.frames import sample_frame_groups
from hma.video.quality import QualityReport, iter_selected_frames

report = QualityReport()
for item in iter_selected_frames(sample_frame_groups("path/to/video.mp4"), report=report):
    print(item.frame.index, round(item.frame.seconds, 2), round(item.quality.sharpness))
print(report.summary())  # filled in as the frames are consumed
```

## Removing repeats

A long static shot gives many near-identical frames. Keeping them all inflates the index and
makes that shot count for more than it should when a video is scored by the share of its
frames that matched. Frames are compared in two steps, cheapest first, and the first frame of
a group of near-duplicates wins:

1. **Thumbnails**, before any embedding is computed: a frame whose 16x16 grayscale thumbnail
   differs from an earlier one by less than `VIDEO_DEDUPE_PIXEL_DIFF_MAX` is dropped without
   ever being embedded.
2. **Embeddings**, for the frames that passed: a frame closer than
   `VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX` (cosine distance) to a kept frame is dropped.

Each frame is compared with every frame kept so far, so duplicates need not be adjacent.

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_DEDUPE_PIXEL_DIFF_MAX` | Mean absolute thumbnail difference (0-255) below which frames repeat | `2.0` |
| `VIDEO_DEDUPE_EMBEDDING_DISTANCE_MAX` | Cosine distance below which two embeddings repeat | `0.02` |

The defaults are conservative: on 196 clips of the series dataset, an embedding distance of 0.02 removes
only the closest quarter of frames 0.17 s apart and about a tenth of frames 1 s apart, and it
is well below the typical distance between clips of one series (median 0.114).

## The pipeline

[pipeline.py](../pipeline.py) runs the steps in order and streams through the video: decoded frames
are handled one slice at a time, and only the frames it keeps (each with its JPEG and embedding) are
held until it returns.

```python
from hma.video.pipeline import PipelineConfig, select_video_frames

def embed(jpegs):  # JPEG bytes in, one embedding vector per frame out
    ...

result = select_video_frames("path/to/video.mp4", embed, PipelineConfig.from_env())
print(result.report.summary())
if result.report.no_usable_frames:
    print("nothing to index in this video")
for frame in result.frames:  # KeptFrame: index, seconds, jpeg, embedding, quality
    print(frame.index, round(frame.seconds, 2), len(frame.jpeg))
```

The embedding function is passed in so the pipeline does not depend on how embeddings are made
(the evaluation gets them from HMA's CLIP signal). It receives the frames as the JPEG bytes they
would be uploaded as, at most `VIDEO_EMBED_BATCH_SIZE` at a time.

**No frame cap by default.** A fixed cap would silently drop frames of long videos, so
`VIDEO_MAX_FRAMES` is unset (or `0`) unless you set it. Embedding cost then grows with the
video's length: at the default 1 s gap, an hour of video is about 3,600 frames. When a cap is
set, the frames left after the quality filter and de-duplication are thinned evenly, keeping
the first and last, so the cap spreads over the video's distinct content and a long static shot
cannot use it up. The cap limits how many frames reach the bank, not the cost of embedding: every
frame is embedded before the rest are thinned away.

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_MAX_FRAMES` | Most frames to keep per video; unset, empty or `0` means no cap | no cap |
| `VIDEO_EMBED_BATCH_SIZE` | Frames sent to the embedding function at once | `32` |
| `VIDEO_JPEG_QUALITY` | Quality (1-100) of the JPEG each kept frame is encoded as | `90` |

`result.report` says what happened to the candidates: slices and frames assessed, junk by
reason, slices dropped, pixel and embedding duplicates, frames thinned to the cap, and frames
kept. A video with nothing usable returns no frames and `report.no_usable_frames` is true, so
the caller can tell the user.

Try the sampling on a video. A 3 s clip with no cut gives three groups of candidate frames, one per
second, and `VIDEO_CANDIDATE_WINDOW_SECONDS=0` leaves just their midpoints, at 0.5 s, 1.5 s and 2.5 s:

```bash
PYTHONPATH=. python - <<'EOF'
from hma.video.frames import sample_frame_groups

for group in sample_frame_groups("path/to/video.mp4"):
    print([(frame.index, round(frame.seconds, 2)) for frame in group])
EOF
```
