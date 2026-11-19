# What the video evaluation measures

Every video is a query against the others in the bank (leave one out), and videos of the same
series are the right answers. Frames are stored with HMA's `clip_float` signal only, and looked up
once at the largest frame distance, with the smaller distances taken from that result. How a video
becomes its frames is in [FRAME_SELECTION.md](FRAME_SELECTION.md).

## The suites

There are three suites, the video counterparts of the image suites: Retrieval ranks the videos that
are found, as top-k does; Match sweeps a threshold, as the threshold suite does; and Pairwise
compares pairs of videos directly, computed here from the frame embeddings instead of with HMA's
compare endpoint. The metrics are mAP and precision/recall.

- **Retrieval.** Each frame of the query is looked up in the bank, and a candidate video scores the
  share of the query's frames that matched one of its frames within a frame distance. Candidates are
  ranked by score, and two orderings are reported: score alone (candidates that tie are in a fixed
  random order), and score then the closest matching frame. The closest frame is taken from
  everything the lookup returned, which reaches to the largest swept distance (`VIDEO_DISTANCE_MAX`),
  not only to the distance being evaluated, so that ordering changes little with the distance and
  depends on `VIDEO_DISTANCE_MAX`. This is swept over frame distances (`VIDEO_DISTANCE_STEP` to
  `VIDEO_DISTANCE_MAX`). The series are split in two with a fixed seed, and the distance that gives the
  best score-only mAP@3 on the train half is reported on the test half, beside the default
  `VIDEO_FRAME_DISTANCE`.
- **Match.** Precision and recall of calling two videos a match, sweeping the score threshold at the
  default frame distance, and again at the distance chosen on the train half, for test queries only.
- **Pairwise.** A distance between two videos from their frame embeddings alone: the mean, over
  one video's frames, of the distance to the closest frame of the other, averaged over both
  directions. It needs no index or threshold, so it measures the embeddings. Its mAP, precision/recall
  and distance distribution are reported.

A video that cannot be read, or has no usable frame, is never dropped from the metrics: it cannot be
found, which counts against the result. In Pairwise it is as far from every video as a distance can be
(2.0, the largest cosine distance).

## Results

| Output file (in the results folder) | Contents |
|---|---|
| `retrieval_sweep_clip_float_results.csv` | mAP@1..`MAX_K` for each frame distance, ranking, and series group (all, train, test) |
| `frame_distance_choice_clip_float_results.csv` | the default distance and the one chosen on train, with their train and test mAP@3 |
| `map_by_series_{score_only,closest_frame}_clip_float_results.csv` | mAP@k per series at the default distance |
| `match_precision_recall_clip_float_results.csv`, `..._curve.png` | precision/recall over the score threshold, and the same at the chosen distance as `match_precision_recall_selected_...` |
| `pairwise_clip_float_compare.csv` | the distance for every pair of videos with frames (paths anonymised unless `DEANONYMIZE_IMAGE_PATHS=1`) |
| `map_by_series_pairwise_...`, `precision_recall_...`, `distance_distribution_...png` | the pairwise metrics |
| `content_id_to_frame_clip_float.json` | which video and frame each bank item is |

## Configuration

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_INPUT_DIR` | Folder of series folders | `hma/video/resources/video-series-dataset/series` |
| `BANK_NAME` | Bank to create in HMA | `VIDEO_TEST_BANK` (`VIDEO_SMOKE_TEST` for the smoke test) |
| `VIDEO_FRAME_DISTANCE` | The default frame distance. 0.10 is HMA's own `clip_float` threshold | `0.10` |
| `VIDEO_DISTANCE_MAX`, `VIDEO_DISTANCE_STEP` | The frame distances swept: step, 2 x step, ... up to the maximum | `0.30`, `0.05` |
| `MAX_K` | Largest k of the mAP@k columns in the sweep (at least 3) | `5` |
| `VIDEO_TEST_FRACTION`, `VIDEO_SPLIT_SEED` | Share of the series held out, and the seed of the split and of the order of tied candidates | `0.5`, `0` |
| `MAX_WORKERS` | Parallel lookups | `8` |

The frame selection has its own variables, in [FRAME_SELECTION.md](FRAME_SELECTION.md).

## Indexing and scoring

[frame_index.py](../frame_index.py) connects the pipeline to HMA. HMA hashes the frames with its
CLIP image signal (`clip_float`), and each kept frame is added to a bank as a precomputed hash,
one bank item per frame. The bank does not remember which video an item came from, so
`index_videos` returns a `VideoIndex` that does (and holds each video's frame embeddings).
A video that cannot be read, or has no usable frame, is listed in `VideoIndex.unusable` and left out.

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

HMA builds its search index in the background, and the index size it reports can be ahead of what
lookups see. After `index_videos`, call `wait_until_searchable(client, index)` from the same module,
which waits until a lookup finds the frame added last, before looking anything up.
