"""Evaluate video semantic hashing through HMA, on a directory of series folders.

Each video is reduced to a bag of frames (hma/video/pipeline.py), every frame is stored in an HMA
bank, and each video is then used as a query against the others (leave one out). Three suites
report mAP and precision/recall; see hma/video/README.md for what each one measures.
"""

import csv
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from hma.hma_client import HMAClient
from hma.logging_utils import _log_info, _log_warning, get_log_file, setup_logging
from hma.metrics.common import classify_pairs_by_series, normalize_pairwise_path
from hma.metrics.distance_distribution import compute_distance_distribution_from_labeled
from hma.metrics.map import compute_map_from_rankings
from hma.metrics.precision_recall import compute_precision_recall_from_labeled
from hma.results_utils import get_results_dir, save_content_id_mapping, write_results
from hma.video import retrieval
from hma.video.frame_index import VideoIndex, index_videos, lookup_frames, wait_until_searchable
from hma.video.pairwise import distance_pairs, video_distances
from hma.video.series_labels_utils import find_video_series, get_video_files

video_input_dir = Path("./hma/video/resources/videos")

# The frame signal is the CLIP image embedding as floats; the integer "clip" signal has no vector to compare.
SIGNAL_TYPE = "clip_float"

# The k of the mAP@k by which the frame distance is chosen on the train series.
SELECT_K = 3


@dataclass(frozen=True)
class EvaluationConfig:
    """How the frame distance is swept and the series are split."""

    frame_distance: float = 0.10  # the headline distance: HMA's own clip_float threshold
    distance_max: float = 0.30  # the sweep covers step, 2 * step ... distance_max
    distance_step: float = 0.05
    max_k: int = 5  # the largest k of the mAP@k columns in the sweep
    test_fraction: float = 0.5  # share of the series held out to report the chosen distance
    seed: int = 0  # seeds the series split and the order of tied candidates
    max_workers: int = 8  # parallel lookups

    def __post_init__(self):
        if self.frame_distance <= 0:
            raise ValueError("frame_distance must be positive")
        if self.max_k < SELECT_K:
            raise ValueError(f"max_k must be at least {SELECT_K}, the k the frame distance is chosen by")
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

        def env(name, default, cast):
            value = os.getenv(name, "").strip()
            return cast(value) if value else default

        return cls(
            frame_distance=env("VIDEO_FRAME_DISTANCE", defaults.frame_distance, float),
            distance_max=env("VIDEO_DISTANCE_MAX", defaults.distance_max, float),
            distance_step=env("VIDEO_DISTANCE_STEP", defaults.distance_step, float),
            max_k=env("MAX_K", defaults.max_k, int),
            test_fraction=env("VIDEO_TEST_FRACTION", defaults.test_fraction, float),
            seed=env("VIDEO_SPLIT_SEED", defaults.seed, int),
            max_workers=env("MAX_WORKERS", defaults.max_workers, int),
        )

    @property
    def grid(self) -> list[float]:
        return retrieval.distance_grid(self.distance_max, self.distance_step)

    @property
    def lookup_distance(self) -> float:
        """The one distance every lookup is made at; smaller distances are filtered from its results."""
        return max(self.grid[-1], self.frame_distance)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _map_columns(maps: Sequence[float]) -> dict[str, str]:
    return {f"map@{k}": f"{value:.6f}" for k, value in enumerate(maps, 1)}


def evaluate(
    client: HMAClient,
    index: VideoIndex,
    videos: Sequence[str],
    series: Mapping[str, Sequence[str]],
    config: EvaluationConfig,
) -> dict:
    """Run the three suites on videos already indexed in the bank and write their results.

    videos are all the videos of the dataset, including any the index could not use: those can
    never be found, which the metrics count against the result. Returns a summary of the choices made.
    """
    names = {video: normalize_pairwise_path(video) for video in videos}
    if len(set(names.values())) != len(names):
        raise ValueError("Two videos have the same normalised path.")
    series_to_videos = {name: {names[v] for v in members} for name, members in series.items()}
    results_dir = get_results_dir()

    _log_info(f"Looking up the frames of {len(index.embeddings)} videos at distance {config.lookup_distance}...")
    started = time.time()
    matches = {
        video: lookup_frames(client, index.embeddings[video], config.lookup_distance, SIGNAL_TYPE, config.max_workers)
        if video in index.embeddings
        else []
        for video in videos
    }
    _log_info(f"Lookups took {time.time() - started:.1f}s")

    chosen = _retrieval_suite(matches, index, videos, names, series_to_videos, config, results_dir)
    _match_suite(matches, index, videos, names, series_to_videos, config, chosen, results_dir)
    _pairwise_suite(index, names, series_to_videos, config, results_dir)
    return chosen


def _retrieval_suite(matches, index, videos, names, series_to_videos, config, results_dir) -> dict:
    """mAP over the frame distances, for both rankings, on all series and on the train and test halves."""
    train, test = retrieval.split_series(series_to_videos, config.test_fraction, config.seed)
    subsets = {"all": set(series_to_videos), "train": train, "test": test}
    cache: dict[float, dict] = {}

    def at(distance: float) -> dict:
        """The rankings and mAP@k at one frame distance."""
        if distance not in cache:
            rankings = retrieval.rankings_at(matches, index.frames, videos, names, distance, config.seed)
            cache[distance] = {
                "rankings": rankings,
                "maps": {
                    (ranking, subset): retrieval.mean_map(
                        {s: series_to_videos[s] for s in members}, ranked, config.max_k
                    )
                    for ranking, ranked in rankings.items()
                    for subset, members in subsets.items()
                },
            }
        return cache[distance]

    rows = [
        {"frame_distance": d, "ranking": ranking, "series": subset, **_map_columns(maps)}
        for d in config.grid
        for (ranking, subset), maps in at(d)["maps"].items()
    ]
    _write_csv(results_dir / f"retrieval_sweep_{SIGNAL_TYPE}_results.csv", rows)

    headline = config.frame_distance
    for ranking, ranked in at(headline)["rankings"].items():
        compute_map_from_rankings(series_to_videos, ranked, str(results_dir / f"map_by_series_{ranking}_{SIGNAL_TYPE}_results.csv"))

    train_key, test_key = f"train_map@{SELECT_K}", f"test_map@{SELECT_K}"

    def selection_map(d: float, subset: str) -> float:
        return at(d)["maps"][(retrieval.SCORE_ONLY, subset)][SELECT_K - 1]

    selected = max(config.grid, key=lambda d: selection_map(d, "train"))
    summary_rows = [
        {
            "choice": choice,
            "frame_distance": d,
            train_key: f"{selection_map(d, 'train'):.6f}",
            test_key: f"{selection_map(d, 'test'):.6f}",
        }
        for choice, d in (("default", headline), ("selected on train", selected))
    ]
    _write_csv(results_dir / f"frame_distance_choice_{SIGNAL_TYPE}_results.csv", summary_rows)

    print(f"  Retrieval: {len(subsets['train'])} train and {len(subsets['test'])} test series")
    for ranking, title in ((retrieval.SCORE_ONLY, "score only"), (retrieval.CLOSEST_FRAME, "score then closest frame")):
        by_distance = ", ".join(f"d={d}: {at(d)['maps'][(ranking, 'all')][SELECT_K - 1]:.3f}" for d in config.grid)
        print(f"  mAP@{SELECT_K}, {title}: {by_distance}")
    for row in summary_rows:
        print(f"  {row['choice']}: d={row['frame_distance']}, train mAP@{SELECT_K} {row[train_key]}, test {row[test_key]}")
    return {"selected_distance": selected, "test_videos": {v for s in subsets["test"] for v in series_to_videos[s]}}


def _match_suite(matches, index, videos, names, series_to_videos, config, chosen, results_dir) -> None:
    """Precision/recall of calling two videos a match, sweeping the score at a fixed frame distance."""
    test_queries = [v for v in videos if names[v] in chosen["test_videos"]]
    runs = (
        ("", config.frame_distance, None),
        ("selected_", chosen["selected_distance"], test_queries),
    )
    for label, distance, queries in runs:
        pairs = retrieval.score_pairs(matches, index.frames, videos, names, distance, queries)
        labeled = classify_pairs_by_series(series_to_videos, pairs)
        csv_path = results_dir / f"match_precision_recall_{label}{SIGNAL_TYPE}_results.csv"
        compute_precision_recall_from_labeled(
            labeled, str(csv_path), str(csv_path).replace("_results.csv", "_curve.png"), SIGNAL_TYPE
        )
        print(f"  Match precision/recall at d={distance}" + (" (test series only)" if queries else "") + f": {csv_path.name}")


def _pairwise_suite(index, names, series_to_videos, config, results_dir) -> None:
    """Distances between videos computed from the embeddings alone: mAP, precision/recall and their distribution."""
    if len(index.embeddings) < 2:
        _log_warning("Fewer than two videos have frames, so there is nothing to compare pairwise.")
        return
    videos, matrix = video_distances(index.embeddings)
    pairs = list(distance_pairs(videos, matrix))
    write_results(
        [
            {"video1": a, "video2": b, "distance": distance, "matched": distance <= config.frame_distance}
            for a, b, distance in pairs
        ],
        f"pairwise_{SIGNAL_TYPE}_compare.csv",
        item_key="video",
    )
    rankings = {
        names[video]: [(names[videos[j]], float(matrix[i, j])) for j in matrix[i].argsort(kind="stable") if j != i]
        for i, video in enumerate(videos)
    }
    compute_map_from_rankings(series_to_videos, rankings, str(results_dir / f"map_by_series_pairwise_{SIGNAL_TYPE}_results.csv"))
    labeled = classify_pairs_by_series(series_to_videos, ((names[a], names[b], d) for a, b, d in pairs))
    compute_precision_recall_from_labeled(
        labeled,
        str(results_dir / f"precision_recall_{SIGNAL_TYPE}_results.csv"),
        str(results_dir / f"precision_recall_{SIGNAL_TYPE}_curve.png"),
        SIGNAL_TYPE,
    )
    compute_distance_distribution_from_labeled(
        labeled, str(results_dir / f"distance_distribution_{SIGNAL_TYPE}.png"), SIGNAL_TYPE
    )
    print(f"  Pairwise: {len(pairs)} video pairs")


def main():
    start_time = time.time()
    setup_logging("test")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["TEST_RUN_TIMESTAMP"] = timestamp
    print("Running video evaluation...")

    video_dir = Path(os.environ.get("VIDEO_INPUT_DIR", str(video_input_dir)))
    videos = get_video_files(video_dir)
    series = find_video_series(video_dir)
    config = EvaluationConfig.from_env()
    print(f"Found {len(videos)} videos in {len(series)} series")

    client = HMAClient()
    client.cleanup_test_environment(SIGNAL_TYPE)
    bank_name = os.getenv("BANK_NAME", "VIDEO_TEST_BANK")
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

    elapsed = int(time.time() - start_time)
    print(f"✓ Video evaluation completed in {elapsed // 60}m {elapsed % 60}s. Results: {get_results_dir()}. Logs: {get_log_file()}")


if __name__ == "__main__":
    main()
