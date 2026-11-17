"""Smoke test for the video evaluation: the happy path on a few generated videos.

Writes the deterministic sample videos of hma/video/sample_videos.py, runs the whole evaluation on
them against a running HMA, and checks the results. In the sample, clips that share a scene have
frames that are identical, so the first hit of every query must be right.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from hma.logging_utils import get_log_file, setup_logging
from hma.results_utils import read_csv_rows
from hma.video.evaluate import SIGNAL_TYPE, EvaluationConfig, run
from hma.video.sample_videos import make_sample_videos

EXPECTED_FILES = [
    "retrieval_sweep_{s}_results.csv",
    "frame_distance_choice_{s}_results.csv",
    "map_by_series_score_only_{s}_results.csv",
    "map_by_series_closest_frame_{s}_results.csv",
    "match_precision_recall_{s}_results.csv",
    "match_precision_recall_{s}_curve.png",
    "match_precision_recall_selected_{s}_results.csv",
    "pairwise_{s}_compare.csv",
    "map_by_series_pairwise_{s}_results.csv",
    "precision_recall_{s}_results.csv",
    "distance_distribution_{s}.png",
    "content_id_to_frame_{s}.json",
]


def check_results(summary: dict, config: EvaluationConfig) -> list[str]:
    """Return what is wrong with the results of a run on the sample videos; empty if all is well."""
    results_dir = summary["results_dir"]
    problems = [f"missing {name}" for name in (n.format(s=SIGNAL_TYPE) for n in EXPECTED_FILES) if not (results_dir / name).exists()]
    if problems:
        return problems
    if summary["unusable"]:
        problems.append(f"{summary['unusable']} sample videos had no usable frame")

    for row in read_csv_rows(results_dir / f"retrieval_sweep_{SIGNAL_TYPE}_results.csv"):
        # Ties are broken by the closest frame, which is identical for the clips that share a scene.
        # Without that tie-break the first hit is right only while no other series is within the distance.
        if row["series"] == "all" and (row["ranking"] == "closest_frame" or float(row["frame_distance"]) <= config.frame_distance):
            if float(row["map@1"]) != 1.0:
                problems.append(f"map@1 is {row['map@1']}, not 1.0, for {row['ranking']} at distance {row['frame_distance']}")

    pairwise = read_csv_rows(results_dir / f"map_by_series_pairwise_{SIGNAL_TYPE}_results.csv")[-1]
    if float(pairwise["map@1"]) != 1.0:
        problems.append(f"pairwise map@1 is {pairwise['map@1']}, not 1.0")

    curve = read_csv_rows(results_dir / f"match_precision_recall_{SIGNAL_TYPE}_results.csv")
    if not any(float(r["precision"]) == 1.0 and float(r["recall"]) >= 0.6 for r in curve):
        problems.append("the match curve has no point with full precision and recall of at least 0.6")
    return problems


def main() -> int:
    setup_logging("smoke")
    os.environ["TEST_RUN_TIMESTAMP"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(os.getenv("OUTPUT_DIR", "./results"))
    sample_dir = output_root / "video_smoke_samples"
    bank_name = os.getenv("BANK_NAME", "VIDEO_SMOKE_TEST")
    config = EvaluationConfig.from_env()

    print("Running video smoke test...")
    print(f"$ python -m hma.video.sample_videos {sample_dir}")
    make_sample_videos(sample_dir)
    print(f"$ VIDEO_INPUT_DIR={sample_dir} BANK_NAME={bank_name} python hma/video/evaluate.py")
    summary = run(sample_dir, bank_name, config)

    problems = check_results(summary, config)
    if problems:
        print("✗ Video smoke test failed:")
        for problem in problems:
            print(f"  - {problem}")
        print(f"Results: {summary['results_dir']}. Logs: {get_log_file()}")
        return 1
    print(f"✓ Video smoke test passed: {summary['videos']} videos, {summary['frames']} frames. Results: {summary['results_dir']}. Logs: {get_log_file()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
