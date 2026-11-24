# What the video evaluation measures

Every video is a query against the others in the bank (leave one out), and videos of the same
series are the right answers. Frames are stored with HMA's `clip_float` signal only. How a video
becomes its frames is in [FRAME_SELECTION.md](FRAME_SELECTION.md).

## How videos are matched

A query asks: *which videos in the bank match this one, best first?*

1. **Look up.** Every frame of the query is looked up in HMA. Two frames match when their cosine
   distance (1 minus cosine similarity) is at most the *frame distance*, 0.10 by default (a
   similarity of 0.90, HMA's own `clip_float` threshold).
2. **Group.** The matched frames are grouped by the video they came from. A query frame counts once
   per video, at the distance of the closest frame it matched there.
3. **Rank.** Each matched frame adds `1 - (distance / frame distance)` to the video's score: 1 for an
   identical frame, 0 for one right at the frame distance. The highest score comes first, and equal scores are
   ordered by name.
4. **Return.** The first k videos are the answer. A video with no matched frame is never returned, so
   the list can be shorter than k. It is empty when nothing in the bank matches, or when the query has no
   usable frame.

For example, with a frame distance of 0.10:

| Video | Distances of the matched frames | Score | Rank |
|---|---|---|---|
| A | 0.04, 0.04, 0.04, 0.04, 0.04 | 5 x 0.6 = 3.0 | 1 |
| B | 0.02, 0.02, 0.02 | 3 x 0.8 = 2.4 | 2 |
| C | 0.01 | 1 x 0.9 = 0.9 | 3 |
| D | 0.09, 0.09, 0.09, 0.09, 0.09 | 5 x 0.1 = 0.5 | 4 |
| E | 0.10 | 1 x 0.0 = 0.0 | 5 |
| F | none within 0.10 | | not returned |

Both the number of matched frames and how close they are count. One very close frame does not outrank several
good ones (C is below A), and several barely matching frames do not outrank fewer close ones (D is below B and
C). Frames that match nothing add nothing, so a short clip is found inside a long video. The frame distance is
part of the score, so changing it can also change the order. The score is not divided by the number of frames,
so compare scores within one query only.

The frame distance and k are the two settings a deployment would tune. This evaluation does not choose
them: the clips of a series are consecutive pieces of one source video, which is not what a deployment would
see. It ranks at one frame distance, reports every k up to the size of the largest series, and shows how the
matches change with the frame distance. HMA's default index is approximate, so a lookup can occasionally miss
a match (see [hma/README.md](../../README.md)).

One accidental frame within the frame distance is enough for a video to be returned. It adds little to the
score when it is near the frame distance, and a smaller frame distance makes it rarer. A longer video has more
frames that could match by accident, but a query frame counts once per video, so its score cannot
exceed the number of query frames.

## The suites

There are three suites, the video counterparts of the image suites: Retrieval ranks the videos that
are found, as top-k does; Match sweeps a threshold, as the threshold suite does; and Pairwise
compares pairs of videos directly, computed here from the frame embeddings instead of with HMA's
compare endpoint. The metrics are mAP and precision/recall.

- **Retrieval.** Every video is a query, and the videos it matches at the frame distance are ranked
  as above. The mAP@k (mean average precision of the first k results; 1.0 is perfect) is averaged over
  the series. A video of the same series that is not returned counts as missed.
- **Match.** Looks at all the videos a query matches, before the list is cut to k. Precision is the share of
  them that are of the query's series, recall is the share of the videos of the query's series that are among
  them, and F1 is high only when both are. A folder with a single video is not a series, and its pairs are not
  counted here or in Pairwise. The three numbers are reported at every frame distance from `VIDEO_DISTANCE_STEP` to `VIDEO_DISTANCE_MAX` (0.05, 0.10,
  ... 0.30) and at `VIDEO_FRAME_DISTANCE`. HMA is asked once per frame, at the largest distance, and a smaller
  distance uses the same result with the farther matches dropped.
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
| `map_by_series_retrieval_clip_float_results.csv` | mAP@k at the frame distance, for each series and their mean, with k up to the size of the largest series |
| `match_precision_recall_clip_float_results.csv`, `..._curve.png` | precision, recall and F1 at each frame distance |
| `pairwise_clip_float_compare.csv` | the distance for every pair of videos with frames (paths anonymised unless `DEANONYMIZE_IMAGE_PATHS=1`) |
| `map_by_series_pairwise_...`, `precision_recall_...`, `distance_distribution_...png` | the pairwise metrics |
| `content_id_to_frame_clip_float.json` | which video and frame each bank item is |

## Which number answers which question

| Question | Read |
|---|---|
| Are the videos of a series returned, and ranked first? | Retrieval mAP@k |
| How do wrong and missed matches change with the frame distance? | Match precision and recall |
| How good are the embeddings alone, with no threshold? | Pairwise mAP and its distance distribution |

## Configuration

| Environment variable | Meaning | Default |
|---|---|---|
| `VIDEO_INPUT_DIR` | Folder of series folders | `hma/video/resources/video-series-dataset/series` |
| `BANK_NAME` | Bank to create in HMA | `VIDEO_TEST_BANK` (`VIDEO_SMOKE_TEST` for the smoke test) |
| `VIDEO_FRAME_DISTANCE` | The frame distance of the ranking. 0.10 is HMA's own `clip_float` threshold | `0.10` |
| `VIDEO_DISTANCE_MAX`, `VIDEO_DISTANCE_STEP` | The frame distances the Match suite reports: step, 2 x step, ... up to the maximum | `0.30`, `0.05` |
| `MAX_WORKERS` | Parallel lookups | `8` |

The frame selection has its own variables, in [FRAME_SELECTION.md](FRAME_SELECTION.md).

## Indexing and matching

[frame_index.py](../frame_index.py) connects the pipeline to HMA. HMA hashes the frames with its
CLIP image signal (`clip_float`), and each kept frame is added to a bank as a precomputed hash,
one bank item per frame. The bank does not remember which video an item came from, so
`index_videos` returns a `VideoIndex` that does (and holds each video's frame embeddings).
A video that cannot be read, or has no usable frame, is listed in `VideoIndex.unusable` and left out.

`lookup_frames` looks up a query's frames and `rank_matches` ranks the videos they matched. The query's own
video is left out with `exclude_video`.

```python
from hma.hma_client import HMAClient
from hma.video.frame_index import index_videos, lookup_frames, rank_matches, wait_until_searchable

client = HMAClient()
client.setup_bank("VIDEOS")
index = index_videos(client, "VIDEOS", ["a.mp4", "b.mp4", "c.mp4"])  # needs a running HMA
wait_until_searchable(client, index)

query = "a.mp4"
matches = lookup_frames(client, index.embeddings[query], max_distance=0.3)
for distance in (0.1, 0.2, 0.3):  # one lookup serves every smaller one
    print(distance, rank_matches(matches, index.frames, distance, exclude_video=query)[:3])  # the top 3
```

HMA builds its search index in the background, and the index size it reports can be ahead of what
lookups see. `wait_until_searchable` waits until a lookup finds the frame added last, so call it after
`index_videos` and before looking anything up.
