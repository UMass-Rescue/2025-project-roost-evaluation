# What the video evaluation measures

Every video is a query against the others in the bank (leave one out), and videos of the same
series are the right answers. Frames are stored with HMA's `clip_float` signal only. How a video
becomes its frames is in [FRAME_SELECTION.md](FRAME_SELECTION.md).

## How two videos are compared

1. **Two frames match** when their cosine distance (1 minus the cosine similarity of their CLIP
   embeddings) is at most the *frame distance*. The default is 0.10, which is a similarity of
   0.90 and is HMA's own `clip_float` threshold.
2. **A candidate video's score** is the share of the query's frames that matched at least one of
   the candidate's frames. If a query has 3 frames and 2 of them found a frame of video B, B scores
   0.67. The query's own video is never scored.
3. **Candidates are ranked** by score, highest first. Candidates with the same score (common, since
   a clip has only a few frames) are ordered by their *closest frame*: the smallest distance between
   any frame of the query and any frame of the candidate. Candidates still tied after that are in
   name order. The closest frame is taken from everything the lookup returned, which reaches to
   `VIDEO_DISTANCE_MAX`, so it can rank a video that matched nothing within the frame distance.

The ranking uses the closest frame only to break ties, not as the score, because one close pair of
frames is a weak signal: a shared logo or similar background can make a single pair of unrelated
videos look close, whereas a share of the frames needs several of them to agree. The closest pair
fits the aim that any part of one video can match any part of another more directly, so this is
a trade-off, and the Pairwise suite below reports a distance-only comparison to set beside it.

### Choosing the frame distance

The best frame distance is not known in advance: too small and videos of the same series find
nothing, too large and other series match. So the evaluation *sweeps* it, trying every value from
`VIDEO_DISTANCE_STEP` up to `VIDEO_DISTANCE_MAX` (0.05, 0.10, ... 0.30). HMA is asked only once per
frame, at the largest distance. Each match comes back with its distance, so the result for a smaller
distance is that result with the farther matches dropped.

For each distance, the Match suite (below) gives an F1: the best balance of precision and recall over
the score thresholds. The distance is chosen on half of the series and reported on the other half,
so the number shows whether a distance chosen on some series holds up on series it never saw.

- The series are shuffled once with a seed (`VIDEO_SPLIT_SEED`) and divided into a train half and a
  test half. The seed is fixed, so the split is the same on every run and runs can be compared.
- The distance with the best F1 on the train half is the *selected* distance. Its F1 and the retrieval
  mAP are reported on the test half, beside the same numbers at the default `VIDEO_FRAME_DISTANCE`.

## The suites

There are three suites, the video counterparts of the image suites: Retrieval ranks the videos that
are found, as top-k does; Match sweeps a threshold, as the threshold suite does; and Pairwise
compares pairs of videos directly, computed here from the frame embeddings instead of with HMA's
compare endpoint. The metrics are mAP and precision/recall.

- **Retrieval.** Every video is a query, and the other videos are ranked as above. The mAP@k
  (mean average precision of the first k results; 1.0 is perfect) is averaged over the series and
  reported on all series, the train half and the test half, at the default and at the selected
  frame distance.
- **Match.** A yes/no decision for every pair of videos: a candidate is called a match for a query when
  its score reaches a score threshold, and a match is right when the two are of the same series.
  Precision is the share of the matches called that are right, recall is the share of the right
  matches that were called, and F1 is high only when both are. The score threshold is swept for the
  precision/recall curve, at the default frame distance, and again at the selected distance for the
  test queries only. This suite is where the frame distance changes the result, so it also picks
  the distance (see above).
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
| `frame_distance_sweep_clip_float_results.csv` | the best F1, with its precision and recall, for each frame distance on the train and the test series |
| `frame_distance_choice_clip_float_results.csv` | the default distance and the one selected on train, with their train and test F1 |
| `retrieval_clip_float_results.csv` | mAP@1..`MAX_K` at the default and the selected distance, for all, train and test series |
| `map_by_series_retrieval_clip_float_results.csv` | mAP@k per series at the selected distance |
| `match_precision_recall_clip_float_results.csv`, `..._curve.png` | precision/recall over the score threshold, and the same at the chosen distance as `match_precision_recall_selected_...` |
| `pairwise_clip_float_compare.csv` | the distance for every pair of videos with frames (paths anonymised unless `DEANONYMIZE_IMAGE_PATHS=1`) |
| `map_by_series_pairwise_...`, `precision_recall_...`, `distance_distribution_...png` | the pairwise metrics |
| `content_id_to_frame_clip_float.json` | which video and frame each bank item is |

## Which number answers which question

| Question | Read |
|---|---|
| How well are the videos of a series ranked first? | Retrieval mAP@k on the test series (about 0.96 on the series dataset) |
| How well does a yes/no match decision work, and at which frame distance? | Match F1, precision and recall in `frame_distance_sweep` and `frame_distance_choice` |
| How good are the embeddings alone, with no threshold? | Pairwise mAP and its distance distribution |

On the series dataset the Retrieval mAP is close to its ceiling and changes little with the frame
distance, because the closest frame already separates the series almost perfectly. It says little about
the distance, which is why the Match F1 chooses it. A harder dataset, with look-alike videos in other
series, would likely show more of a difference.

## Configuration

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_INPUT_DIR` | Folder of series folders | `hma/video/resources/video-series-dataset/series` |
| `BANK_NAME` | Bank to create in HMA | `VIDEO_TEST_BANK` (`VIDEO_SMOKE_TEST` for the smoke test) |
| `VIDEO_FRAME_DISTANCE` | The default frame distance. 0.10 is HMA's own `clip_float` threshold | `0.10` |
| `VIDEO_DISTANCE_MAX`, `VIDEO_DISTANCE_STEP` | The frame distances swept: step, 2 x step, ... up to the maximum | `0.30`, `0.05` |
| `MAX_K` | Largest k of the mAP@k columns (at least 3) | `5` |
| `VIDEO_TEST_FRACTION`, `VIDEO_SPLIT_SEED` | Share of the series held out, and the seed of the split | `0.5`, `0` |
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
