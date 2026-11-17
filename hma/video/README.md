# Video evaluation

Video is evaluated as a bag of frames: representative frames are chosen from each video,
hashed with the existing CLIP image signal, and stored in HMA. A video then matches another in
proportion to how many of its frames match one of the other's.

Where this README refers to "the series dataset" it means UMass Rescue Lab's
[video-series-dataset](https://github.com/UMass-Rescue/video-series-dataset): 398 clips of
3 seconds each (18 frames at 6 fps), cut from the first 100 videos of subset 51 of Meta's
SA-V dataset and grouped into 100 series, each series being the consecutive segments of one
source video.

## Quick start

Start HMA first (the root [README.md](../../README.md)'s Quick Start). Each run **resets HMA's
database**, as the image evaluation does, so it replaces whatever other evaluations left in it.

```bash
make video-smoke-test         # about 1 minute, no dataset needed: see "Smoke test" below
make fetch-video-dataset      # once: the series dataset, about 550 MB, at a pinned commit
make run-video-evaluation     # the full evaluation, about 4 minutes on the series dataset
```

The same in Docker, which already sees the repo (and so the dataset) at `/build`:

```bash
docker compose run --rm evaluation python hma/video/smoke_test.py
docker compose run --rm evaluation python hma/video/evaluate.py
```

Results are written to `results/evaluation_results/<timestamp>/` (`OUTPUT_DIR` changes the root)
and the log to `results/test_run_logs/`. To point the evaluation at another folder of series
folders, set `VIDEO_INPUT_DIR` to it (a series is a subfolder with at least two videos).

### Smoke test

[smoke_test.py](smoke_test.py) writes 12 generated clips ([sample_videos.py](sample_videos.py): four
series of three clips, the same files every time), runs the whole evaluation on them, and checks
that every result file exists and that the first hit of every query is right. It prints the
commands that reproduce it, and exits non-zero if a check fails. Expect
`✓ Video smoke test passed: 12 videos, 24 frames`. The generated clips are in
`results/video_smoke_samples/`, which can be deleted afterwards; to write them elsewhere run
`python -m hma.video.sample_videos <folder>`.

## What it measures

Every video is a query against the others in the bank (leave one out), and videos of the same
series are the right answers. There are three suites; the metrics are mAP and precision/recall.

- **Retrieval.** Each frame of the query is looked up in the bank, and a candidate video scores the
  share of the query's frames that matched one of its frames within a frame distance. Candidates are
  ranked by score, and two orderings are reported: score alone (candidates that tie are in a fixed
  random order), and score then the closest matching frame. This is swept over frame distances
  (`VIDEO_DISTANCE_STEP` to `VIDEO_DISTANCE_MAX`). The series are split in two with a fixed seed, and
  the distance that gives the best score-only mAP@3 on the train half is reported on the test half,
  beside the default `VIDEO_FRAME_DISTANCE`.
- **Match.** Precision and recall of calling two videos a match, sweeping the score threshold at the
  default frame distance, and again at the distance chosen on the train half, for test queries only.
- **Pairwise.** A distance between two videos from their frame embeddings alone: the mean, over
  one video's frames, of the distance to the closest frame of the other, averaged over both
  directions. It needs no index or threshold, so it measures the embeddings. Its mAP, precision/recall
  and distance distribution are reported.

A video with no usable frame is never dropped from the metrics: it cannot be found, which counts
against the result.

| Output file (in the results folder) | Contents |
|---|---|
| `retrieval_sweep_clip_float_results.csv` | mAP@1..`MAX_K` for each frame distance, ranking, and series group (all, train, test) |
| `frame_distance_choice_clip_float_results.csv` | the default distance and the one chosen on train, with their train and test mAP@3 |
| `map_by_series_{score_only,closest_frame}_clip_float_results.csv` | mAP@k per series at the default distance |
| `match_precision_recall_clip_float_results.csv`, `..._curve.png` | precision/recall over the score threshold, and the same at the chosen distance as `match_precision_recall_selected_...` |
| `pairwise_clip_float_compare.csv` | the distance for every pair of videos (paths anonymised unless `DEANONYMIZE_IMAGE_PATHS=1`) |
| `map_by_series_pairwise_...`, `precision_recall_...`, `distance_distribution_...png` | the pairwise metrics |
| `content_id_to_frame_clip_float.json` | which video and frame each bank item is |

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_INPUT_DIR` | Folder of series folders | `hma/video/resources/video-series-dataset/series` |
| `BANK_NAME` | Bank to create in HMA | `VIDEO_TEST_BANK` (`VIDEO_SMOKE_TEST` for the smoke test) |
| `VIDEO_FRAME_DISTANCE` | The default frame distance. 0.10 is HMA's own `clip_float` threshold | `0.10` |
| `VIDEO_DISTANCE_MAX`, `VIDEO_DISTANCE_STEP` | The frame distances swept: step, 2 x step, ... up to the maximum | `0.30`, `0.05` |
| `MAX_K` | Largest k of the mAP@k columns in the sweep (at least 3) | `5` |
| `VIDEO_TEST_FRACTION`, `VIDEO_SPLIT_SEED` | Share of the series held out, and the seed of the split and of the order of tied candidates | `0.5`, `0` |
| `MAX_WORKERS` | Parallel lookups | `8` |

The frame selection has its own variables, below.

Video is stored with HMA's `clip_float` signal only. Frames are looked up once at the largest
frame distance and the smaller ones are taken from that result.

## Frame selection

These steps turn a video into the frames to hash: choosing candidate frames ([frames.py](frames.py)),
filtering out low-information ones ([quality.py](quality.py)), removing repeats
([dedupe.py](dedupe.py)), and the [pipeline](pipeline.py) that runs them in order.
[frame_index.py](frame_index.py) then stores the frames in an HMA bank and scores a query video
against it, and [retrieval.py](retrieval.py) and [pairwise.py](pairwise.py) turn those scores and
distances into the rankings the metrics use.

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
from hma.video.quality import select_frames

selected, report = select_frames(sample_frame_groups("path/to/video.mp4"))
print(report.summary())
for item in selected:
    print(item.frame.index, round(item.frame.seconds, 2), round(item.quality.sharpness))
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

[pipeline.py](pipeline.py) runs the steps in order and streams through the video, so a long
one is never held in memory:

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
cannot use it up.

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_MAX_FRAMES` | Most frames to keep per video; unset, empty or `0` means no cap | no cap |
| `VIDEO_EMBED_BATCH_SIZE` | Frames sent to the embedding function at once | `32` |
| `VIDEO_JPEG_QUALITY` | Quality (1-100) of the JPEG each kept frame is encoded as | `90` |

`result.report` says what happened to the candidates: slices and frames assessed, junk by
reason, slices dropped, pixel and embedding duplicates, frames thinned to the cap, and frames
kept. A video with nothing usable returns no frames and `report.no_usable_frames` is true, so
the caller can tell the user.

Try the sampling on a video (a 3 s clip with no cut gives three frames, at 0.5 s, 1.5 s and 2.5 s):

```bash
PYTHONPATH=. python - <<'EOF'
from hma.video.frames import sample_frames

for frame in sample_frames("path/to/video.mp4"):
    print(frame.index, round(frame.seconds, 2), frame.image.shape)
EOF
```

## Indexing and scoring

[frame_index.py](frame_index.py) connects the pipeline to HMA. HMA hashes the frames with its
CLIP image signal (`clip_float`), and each kept frame is added to a bank as a precomputed hash,
one bank item per frame. The bank does not remember which video an item came from, so
`index_videos` returns a `VideoIndex` that does (and holds each video's frame embeddings).
A video with no usable frame is listed in `VideoIndex.unusable` and left out.

A query video is scored against every other video: each of its frames is looked up, and a
candidate's score is the share of the query's frames that matched at least one of the
candidate's frames within a distance. The query's own video is never scored.

```python
from hma.hma_client import HMAClient
from hma.video.frame_index import index_videos, lookup_frames, video_scores

client = HMAClient()
client.setup_bank("VIDEOS")
index = index_videos(client, "VIDEOS", ["a.mp4", "b.mp4", "c.mp4"])  # needs a running HMA

query = "a.mp4"
matches = lookup_frames(client, index.embeddings[query], max_distance=0.3)
for distance in (0.1, 0.2, 0.3):  # one lookup serves every smaller distance
    print(distance, video_scores(matches, index.frames, distance, exclude_video=query))
```

HMA builds its search index in the background, so wait for it to include the new frames
(`HMAClient.wait_for_index_update`) before looking anything up.

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
PYTHONPATH=. python -m pytest hma/unit_tests/ -k video -v
```
