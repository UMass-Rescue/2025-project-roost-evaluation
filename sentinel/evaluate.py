"""
Evaluation script for Sentinel's grooming classifier API.

Flow (EVAL_MODE=test):
1. Stage the 1-line conversations into bank-creation input folders
   (sentinel.dataset.stage_1l_banks).
2. Create + load banks via POST /banks/create (recreated fresh every run).
3. Load the 10-line human-reviewed test set with ground-truth labels
   (sentinel.dataset.load_10l_eval_set).
4. Score each test conversation via POST /score, collecting
   (rare_class_affinity_score, ground_truth_label) pairs.
5. Compute a precision/recall threshold sweep (sentinel.metrics) and save
   CSV + plot results.

EVAL_MODE=smoke (default) runs the same flow against a tiny subset, as a
fast end-to-end sanity check.

Run via (from the repo root, inside the evaluation container):
    docker compose run --rm evaluation python sentinel/evaluate.py
    docker compose run --rm -e EVAL_MODE=test evaluation python sentinel/evaluate.py
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from sentinel.dataset import Conversation, load_10l_eval_set, stage_1l_banks
from sentinel.metrics import compute_precision_recall

SENTINEL_HOST = os.getenv("SENTINEL_HOST", "localhost")
SENTINEL_PORT = os.getenv("SENTINEL_PORT", "8001")
SENTINEL_APP_URL = f"http://{SENTINEL_HOST}:{SENTINEL_PORT}"

DATA_DIR = os.getenv("SENTINEL_DATA_DIR", "/data")
BANK_MODEL_NAME = os.getenv("SENTINEL_BANK_MODEL", "all-MiniLM-L6-v2")

# Bank creation embeds every staged file; generous timeout for CPU-only runs.
CREATE_BANKS_TIMEOUT_SECONDS = int(os.getenv("SENTINEL_CREATE_BANKS_TIMEOUT", "1800"))
SCORE_TIMEOUT_SECONDS = int(os.getenv("SENTINEL_SCORE_TIMEOUT", "60"))

SMOKE_TEST_FILES_PER_CLASS = 10

OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "./results"))

logger = logging.getLogger("sentinel_evaluation")


class SentinelEvaluator:
    """Thin HTTP client for the Sentinel API's health/bank/scoring endpoints."""

    def __init__(self, base_url: str = SENTINEL_APP_URL):
        self.base_url = base_url

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def bank_status(self) -> dict:
        response = requests.get(f"{self.base_url}/banks/status")
        response.raise_for_status()
        return response.json()

    def create_banks(
        self,
        positive_folder: str,
        negative_folder: str,
        output_path: str,
        model_name: str = BANK_MODEL_NAME,
    ) -> dict:
        """Create (and load) banks from folders of text files.

        positive_folder/negative_folder/output_path must be paths that
        resolve inside the sentinel-app container -- see stage_1l_banks().
        """
        payload = {
            "positive_folder": positive_folder,
            "negative_folder": negative_folder,
            "output_path": output_path,
            "model_name": model_name,
        }
        response = requests.post(
            f"{self.base_url}/banks/create",
            json=payload,
            timeout=CREATE_BANKS_TIMEOUT_SECONDS,
        )
        if not response.ok:
            raise RuntimeError(
                f"Failed to create banks: {response.status_code} - {response.text}"
            )
        return response.json()

    def score(
        self,
        texts: List[str],
        top_k: Optional[int] = None,
        min_score_to_consider: Optional[float] = None,
    ) -> dict:
        payload = {"texts": texts}
        if top_k is not None:
            payload["top_k"] = top_k
        if min_score_to_consider is not None:
            payload["min_score_to_consider"] = min_score_to_consider
        response = requests.post(
            f"{self.base_url}/score", json=payload, timeout=SCORE_TIMEOUT_SECONDS
        )
        if not response.ok:
            raise RuntimeError(
                f"Failed to score conversation: {response.status_code} - {response.text}"
            )
        return response.json()


def _score_conversations(
    evaluator: SentinelEvaluator, conversations_with_labels: List[Tuple[Conversation, str]]
) -> List[Tuple[float, str]]:
    """Score each conversation, returning (score, ground_truth_label) pairs."""
    labeled_scores = []
    total = len(conversations_with_labels)
    for i, (conversation, label) in enumerate(conversations_with_labels, 1):
        lines = conversation.lines()
        if not lines:
            logger.warning(f"Skipping empty conversation: {conversation.path}")
            continue
        result = evaluator.score(lines)
        score = result["rare_class_affinity_score"]
        labeled_scores.append((score, label))
        logger.debug(
            f"[{i}/{total}] {conversation.path.name}: score={score:.4f} label={label}"
        )
    return labeled_scores


def run_smoke_test() -> None:
    print("Running Sentinel smoke test...")
    evaluator = SentinelEvaluator()

    print("  Checking health...")
    print(f"    {evaluator.health()}")

    print(f"  Staging a tiny subset ({SMOKE_TEST_FILES_PER_CLASS} files per class)...")
    positive_dir, negative_dir = stage_1l_banks(limit=SMOKE_TEST_FILES_PER_CLASS)
    print(
        f"    positive: {len(list(positive_dir.iterdir()))} files, "
        f"negative: {len(list(negative_dir.iterdir()))} files"
    )

    print("  Creating banks...")
    create_result = evaluator.create_banks(
        str(positive_dir), str(negative_dir), f"{DATA_DIR}/banks/smoke_test"
    )
    print(f"    {create_result}")

    print("  Scoring a sample test conversation...")
    conversations = load_10l_eval_set()
    if not conversations:
        print("  No test conversations with ground truth available, skipping.")
        return
    conversation, label = conversations[0]
    result = evaluator.score(conversation.lines())
    print(
        f"    conversation={conversation.path.name} ground_truth={label} "
        f"score={result['rare_class_affinity_score']:.4f}"
    )
    print("\u2713 Smoke test completed.")


def run_full_evaluation() -> None:
    start_time = time.time()
    print("Running full Sentinel evaluation...")
    evaluator = SentinelEvaluator()

    print("Staging 1-line banks...")
    positive_dir, negative_dir = stage_1l_banks()
    print(
        f"  positive: {len(list(positive_dir.iterdir()))} files, "
        f"negative: {len(list(negative_dir.iterdir()))} files"
    )

    print("Creating banks (this can take a while on CPU)...")
    create_result = evaluator.create_banks(
        str(positive_dir), str(negative_dir), f"{DATA_DIR}/banks/eval_banks"
    )
    print(f"  {create_result}")

    print("Loading 10-line test set...")
    conversations = load_10l_eval_set()
    print(f"  {len(conversations)} conversations with ground truth")

    print("Scoring test conversations...")
    labeled_scores = _score_conversations(evaluator, conversations)
    print(f"  Scored {len(labeled_scores)} conversations")

    print("Computing precision/recall...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = OUTPUT_ROOT / "sentinel_evaluation_results" / timestamp
    output_csv = results_dir / "precision_recall_results.csv"
    output_plot = results_dir / "precision_recall_curve.png"
    compute_precision_recall(labeled_scores, str(output_csv), str(output_plot))
    print(f"  Saved: {output_csv}")
    print(f"  Saved: {output_plot}")

    elapsed = time.time() - start_time
    print(f"\u23f1  Total time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    eval_mode = os.environ.get("EVAL_MODE", "smoke")
    if eval_mode == "smoke":
        run_smoke_test()
    else:
        run_full_evaluation()


if __name__ == "__main__":
    main()
