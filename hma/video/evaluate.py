"""Evaluate video semantic hashing through HMA, on a directory of series folders.

Each video is reduced to a bag of frames (hma/video/pipeline.py), every frame is stored in an HMA
bank, and each video is then used as a query against the others (leave one out). Three suites
report mAP and precision/recall; see hma/video/docs/EVALUATION_DESCRIPTION.md for what each one measures.
"""

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

from hma.hma_client import HMAClient
from hma.logging_utils import _log_info, _log_warning, get_log_file, setup_logging
from hma.metrics.common import classify_pairs_by_series
from hma.metrics.distance_distribution import compute_distance_distribution_from_labeled
from hma.metrics.map import compute_map_from_rankings
from hma.metrics.precision_recall import compute_precision_recall_from_labeled
from hma.results_utils import get_results_dir, save_content_id_mapping, write_csv_rows, write_results
from hma.video import retrieval
from hma.video.env import env_value
from hma.video.frame_index import VideoIndex, index_videos, lookup_frames, wait_until_searchable
from hma.video.pairwise import distance_pairs, video_distances
from hma.video.series_labels_utils import find_video_series, get_video_files

# where hma/video/resources/fetch_dataset.sh puts the series dataset
video_input_dir = Path("./hma/video/resources/video-series-dataset/series")

# The frame signal is the CLIP image embedding as floats; the integer "clip" signal has no vector to compare.
SIGNAL_TYPE = "clip_float"

# The k of the mAP@k printed in the run summary.
SELECT_K = 3


@dataclass(frozen=True)
class EvaluationConfig:
    """How the frame distance is swept and the series are split."""

    frame_distance: float = 0.10  # the headline distance: HMA's own clip_float threshold
    distance_max: float = 0.30  # the sweep covers step, 2 * step ... distance_max
    distance_step: float = 0.05
    max_k: int = 5  # the largest k of the mAP@k columns in the sweep
    test_fraction: float = 0.5  # share of the series held out to report the chosen distance
    seed: int = 0  # seeds the split of the series into train and test
    max_workers: int = 8  # parallel lookups

    def __post_init__(self):
        if self.frame_distance <= 0:
            raise ValueError("frame_distance must be positive")
        if self.max_k < SELECT_K:
            raise ValueError(f"max_k must be at least {SELECT_K}, the k of the summary")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        retrieval.distance_grid(self.distance_max, self.distance_step)  # validates both
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Read VIDEO_FRAME_DISTANCE, VIDEO_DISTANCE_MAX, VIDEO_DISTANCE_STEP, MAX_K,
        VIDEO_TEST_FRACTION, VIDEO_SPLIT_SEED and MAX_WORKERS; unset or empty values keep the defaults."""
        defaults = cls()
        return cls(
            frame_distance=env_value("VIDEO_FRAME_DISTANCE", defaults.frame_distance),
            distance_max=env_value("VIDEO_DISTANCE_MAX", defaults.distance_max),
            distance_step=env_value("VIDEO_DISTANCE_STEP", defaults.distance_step),
            max_k=env_value("MAX_K", defaults.max_k, int),
            test_fraction=env_value("VIDEO_TEST_FRACTION", defaults.test_fraction),
            seed=env_value("VIDEO_SPLIT_SEED", defaults.seed, int),
            max_workers=env_value("MAX_WORKERS", defaults.max_workers, int),
        )

    @property
    def grid(self) -> list[float]:
        return retrieval.distance_grid(self.distance_max, self.distance_step)

    @property
    def lookup_distance(self) -> float:
        """The one distance every lookup is made at; smaller distances are filtered from its results."""
        return max(self.grid[-1], self.frame_distance)


class Selection(NamedTuple):
    """The frame distance chosen on the train series, the videos of the held-out test series, and the split."""

    distance: float
    test_videos: set[str]
    train: set[str]
    test: set[str]


@dataclass(frozen=True)
class _Data:
    """What the suites share: the dataset, what was indexed, the lookups made, and where results go."""

    videos: Sequence[str]
    series_to_videos: Mapping[str, set[str]]
    index: VideoIndex
    matches: Mapping[str, list]
    config: EvaluationConfig
    results_dir: Path


def _map_columns(maps: Sequence[float]) -> dict[str, str]:
    return {f"map@{k}": f"{value:.6f}" for k, value in enumerate(maps, 1)}


def evaluate(
    client: HMAClient,
    index: VideoIndex,
    videos: Sequence[str],
    series: Mapping[str, Sequence[str]],
    config: EvaluationConfig,
) -> Selection:
    """Run the three suites on videos already indexed in the bank and write their results.

    videos are all the videos of the dataset, including any the index could not use: those can
    never be found, which the metrics count against the result. Videos are told apart by the
    strings in videos and series, which must be the same for the same video.
    """
    _log_info(f"Looking up the frames of {len(index.embeddings)} videos at distance {config.lookup_distance}...")
    started = time.time()
    matches = {
        video: lookup_frames(client, index.embeddings[video], config.lookup_distance, SIGNAL_TYPE, config.max_workers)
        if video in index.embeddings
        else []
        for video in videos
    }
    _log_info(f"Lookups took {time.time() - started:.1f}s")

    data = _Data(
        videos=videos,
        series_to_videos={name: set(members) for name, members in series.items()},
        index=index,
        matches=matches,
        config=config,
        results_dir=get_results_dir(),
    )
    train, test = retrieval.split_series(data.series_to_videos, config.test_fraction, config.seed)
    selection = _select_distance(data, train, test)
    _retrieval_suite(data, selection)
    _match_suite(data, selection)
    _pairwise_suite(data)
    return selection


def _select_distance(data: _Data, train: set[str], test: set[str]) -> Selection:
    """Choose the frame distance by the best match F1 on the train series; report it on the test series.

    The match F1 is the best F1 over the score thresholds of calling two videos a match, so it
    is high only when the distance finds the same-series videos without finding others.
    """
    config = data.config
    groups = {"train": train, "test": test}
    queries = {
        group: [v for v in data.videos if v in {m for s in members for m in data.series_to_videos[s]}]
        for group, members in groups.items()
    }

    f1 = {}  # frame distance -> group -> BestF1
    for distance in dict.fromkeys([*config.grid, config.frame_distance]):
        f1[distance] = {
            group: retrieval.best_f1(
                classify_pairs_by_series(
                    data.series_to_videos,
                    retrieval.score_pairs(data.matches, data.index.frames, data.videos, distance, group_queries),
                    normalize=str,
                )
            )
            for group, group_queries in queries.items()
        }
    write_csv_rows(
        data.results_dir / f"frame_distance_sweep_{SIGNAL_TYPE}_results.csv",
        [
            {"frame_distance": d, "series": group, "f1": f"{r.f1:.6f}", "precision": f"{r.precision:.6f}", "recall": f"{r.recall:.6f}"}
            for d in config.grid
            for group, r in f1[d].items()
        ],
    )
    selected = max(config.grid, key=lambda d: f1[d]["train"].f1)
    choices = [
        {"choice": choice, "frame_distance": d, "train_f1": f"{f1[d]['train'].f1:.6f}", "test_f1": f"{f1[d]['test'].f1:.6f}"}
        for choice, d in (("default", config.frame_distance), ("selected on train", selected))
    ]
    write_csv_rows(data.results_dir / f"frame_distance_choice_{SIGNAL_TYPE}_results.csv", choices)
    print(f"  Frame distance chosen on {len(train)} train series (reported on {len(test)} test series), by match F1:")
    for row in choices:
        print(f"  {row['choice']}: d={row['frame_distance']}, train F1 {row['train_f1']}, test {row['test_f1']}")
    return Selection(selected, {v for s in test for v in data.series_to_videos[s]}, train, test)


def _retrieval_suite(data: _Data, selection: Selection) -> None:
    """mAP of ranking the videos by score and then closest frame, at the default and the selected frame distance.

    The per-series table is at the selected distance (the last ranking made).
    """
    config = data.config
    subsets = {
        "all": set(data.series_to_videos),
        "train": selection.train,
        "test": selection.test,
    }
    rows = []
    for choice, distance in (("default", config.frame_distance), ("selected on train", selection.distance)):
        ranked = retrieval.rankings_at(data.matches, data.index.frames, data.videos, distance)
        for group, members in subsets.items():
            maps = retrieval.mean_map({s: data.series_to_videos[s] for s in members}, ranked, config.max_k)
            rows.append({"choice": choice, "frame_distance": distance, "series": group, **_map_columns(maps)})
    write_csv_rows(data.results_dir / f"retrieval_{SIGNAL_TYPE}_results.csv", rows)
    compute_map_from_rankings(
        data.series_to_videos, ranked, str(data.results_dir / f"map_by_series_retrieval_{SIGNAL_TYPE}_results.csv")
    )
    for row in rows:
        if row["series"] != "all":
            print(f"  Retrieval mAP@{SELECT_K}, {row['choice']} (d={row['frame_distance']}), {row['series']} series: {row[f'map@{SELECT_K}']}")


def _match_suite(data: _Data, selection: Selection) -> None:
    """Precision/recall of calling two videos a match, sweeping the score at a fixed frame distance."""
    test_queries = [v for v in data.videos if v in selection.test_videos]
    runs = (("", data.config.frame_distance, None), ("selected_", selection.distance, test_queries))
    for label, distance, queries in runs:
        pairs = retrieval.score_pairs(data.matches, data.index.frames, data.videos, distance, queries)
        labeled = classify_pairs_by_series(data.series_to_videos, pairs, normalize=str)
        csv_path = data.results_dir / f"match_precision_recall_{label}{SIGNAL_TYPE}_results.csv"
        compute_precision_recall_from_labeled(
            labeled, str(csv_path), str(csv_path).replace("_results.csv", "_curve.png"), SIGNAL_TYPE
        )
        print(f"  Match precision/recall at d={distance}" + (" (test series only)" if queries else "") + f": {csv_path.name}")


# The largest cosine distance. A video with no frames is as far from every video as it can be.
FARTHEST = 2.0


def _pairs_with_unusable(videos: Sequence[str], index: VideoIndex) -> list[tuple[str, str, float]]:
    """Each pair that has a video with no frames in it, once, at the farthest distance."""
    unusable = [v for v in videos if v not in index.embeddings]
    return [
        (video, other, FARTHEST)
        for i, video in enumerate(unusable)
        for other in videos
        if other != video and other not in unusable[:i]
    ]


def _pairwise_suite(data: _Data) -> None:
    """Distances between videos computed from the embeddings alone: mAP, precision/recall and their distribution."""
    if len(data.index.embeddings) < 2:
        _log_warning("Fewer than two videos have frames, so there is nothing to compare pairwise.")
        return
    videos, matrix = video_distances(data.index.embeddings)
    pairs = list(distance_pairs(videos, matrix))
    write_results(
        [
            {"video1": a, "video2": b, "distance": distance, "matched": distance <= data.config.frame_distance}
            for a, b, distance in pairs
        ],
        f"pairwise_{SIGNAL_TYPE}_compare.csv",
        item_key="video",
    )
    rankings = {
        video: [(videos[j], float(matrix[i, j])) for j in matrix[i].argsort(kind="stable") if j != i]
        for i, video in enumerate(videos)
    }
    compute_map_from_rankings(
        data.series_to_videos, rankings, str(data.results_dir / f"map_by_series_pairwise_{SIGNAL_TYPE}_results.csv")
    )
    labeled = classify_pairs_by_series(
        data.series_to_videos, [*pairs, *_pairs_with_unusable(data.videos, data.index)], normalize=str
    )
    compute_precision_recall_from_labeled(
        labeled,
        str(data.results_dir / f"precision_recall_{SIGNAL_TYPE}_results.csv"),
        str(data.results_dir / f"precision_recall_{SIGNAL_TYPE}_curve.png"),
        SIGNAL_TYPE,
    )
    compute_distance_distribution_from_labeled(
        labeled, str(data.results_dir / f"distance_distribution_{SIGNAL_TYPE}.png"), SIGNAL_TYPE
    )
    print(f"  Pairwise: {len(pairs)} video pairs")


def run(video_dir: Path, bank_name: str, config: EvaluationConfig) -> dict:
    """Index the videos of video_dir in a fresh bank and evaluate them, writing the results.

    Resets the HMA database first, as the image evaluation does. Returns the results folder and
    how many videos and frames were indexed.
    """
    videos = get_video_files(video_dir)
    series = find_video_series(video_dir)
    print(f"Found {len(videos)} videos in {len(series)} series")

    client = HMAClient()
    client.cleanup_test_environment(SIGNAL_TYPE)
    if not client.setup_bank(bank_name):
        raise RuntimeError(f"Could not set up bank {bank_name}")

    print("Selecting frames and adding them to the bank...")
    index = index_videos(client, bank_name, videos, SIGNAL_TYPE)
    frame_map = {content_id: asdict(ref) for content_id, ref in index.frames.items()}
    _log_info(f"Saved frame mapping to {save_content_id_mapping(frame_map, SIGNAL_TYPE, name='content_id_to_frame')}")
    print(f"Added {len(index.frames)} frames from {len(index.embeddings)} videos")
    if index.unusable:
        print(f"WARNING: {len(index.unusable)} videos had no usable frame and cannot be found: see the log")

    print("Waiting for HMA to refresh its index...")
    wait_until_searchable(client, index, SIGNAL_TYPE)
    evaluate(client, index, videos, series, config)
    return {
        "results_dir": get_results_dir(),
        "videos": len(videos),
        "frames": len(index.frames),
        "unusable": len(index.unusable),
    }


def main():
    start_time = time.time()
    setup_logging("test")
    os.environ["TEST_RUN_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Running video evaluation...")
    summary = run(
        Path(os.environ.get("VIDEO_INPUT_DIR", str(video_input_dir))),
        os.getenv("BANK_NAME", "VIDEO_TEST_BANK"),
        EvaluationConfig.from_env(),
    )
    elapsed = int(time.time() - start_time)
    print(f"✓ Video evaluation completed in {elapsed // 60}m {elapsed % 60}s. Results: {summary['results_dir']}. Logs: {get_log_file()}")


if __name__ == "__main__":
    main()
