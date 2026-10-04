import os
import pickle
import argparse
from typing import List, Set, Dict, Tuple, Any
import numpy as np
import pandas as pd

from utils import DEFAULT_TOP_K_CHILDREN, DEFAULT_TOP_N_RERANK

DEFAULT_DATASET_PKL = "golden_dataset.pkl"
DEFAULT_OUTPUT_CSV = "retreiver_evaluation_summary.csv"


# ---------------------------------------------------------------------------
# Core Ranking & Retrieval Metrics
# ---------------------------------------------------------------------------
def get_doc_ids(docs: List[Any]) -> List[str]:
    """Extract doc_id from a list of LangChain Document objects or dicts."""
    ids = []
    for d in docs:
        if hasattr(d, "metadata"):
            ids.append(d.metadata.get("doc_id"))
        elif isinstance(d, dict):
            ids.append(d.get("metadata", {}).get("doc_id") or d.get("doc_id"))
    return [i for i in ids if i is not None]


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Precision@K = (relevant retrieved in top K) / K"""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Recall@K = (relevant retrieved in top K) / (total relevant)"""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates Reciprocal Rank (RR) = 1 / rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates Average Precision (AP) across all ranks."""
    if len(relevant_ids) == 0:
        return 0.0
    hits = 0
    precisions = []
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant_ids)


def dcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Discounted Cumulative Gain at rank K."""
    top_k = retrieved_ids[:k]
    return sum(
        (1.0 if doc_id in relevant_ids else 0.0) / np.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
    )


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / idcg


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def evaluate_retriever(
    golden_dataset: List[Dict[str, Any]],
    context_key: str,
    k_values: Tuple[int, ...] = (1, 3, 5)
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluates a specific retriever's context key across all dataset questions.
    Returns (per_query_dataframe, averaged_metrics_dict).
    """
    per_query_rows = []

    for item in golden_dataset:
        relevant_ids = set(item.get("relevant_docs", []))
        retrieved_docs = item.get(context_key, [])
        retrieved_ids = get_doc_ids(retrieved_docs)
        has_relevant = len(relevant_ids) > 0

        row = {
            "question_id": item.get("question_id"),
            "num_relevant": len(relevant_ids),
            "num_retrieved": len(retrieved_ids),
            "MRR": reciprocal_rank(retrieved_ids, relevant_ids) if has_relevant else np.nan,
            "MAP": average_precision(retrieved_ids, relevant_ids) if has_relevant else np.nan,
        }

        for k in k_values:
            row[f"Precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row[f"Recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k) if has_relevant else np.nan
            row[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k) if has_relevant else np.nan

        per_query_rows.append(row)

    df = pd.DataFrame(per_query_rows)
    metric_cols = [c for c in df.columns if c not in ("question_id", "num_relevant", "num_retrieved")]
    averaged = df[metric_cols].mean(skipna=True).to_dict()

    return df, averaged


# ---------------------------------------------------------------------------
# Optional: Live Re-Retrieval Runner
# ---------------------------------------------------------------------------
def run_live_retrieval(golden_dataset: List[Dict[str, Any]]):
    """
    Executes live retrieval with the current vectorstore and advanced pipeline
    to refresh retrieved contexts for all questions in the golden dataset.
    """
    print("\n--- Running Live Retrieval Pipeline ---")
    from rag_pipeline import get_pipeline
    pipeline = get_pipeline()

    for idx, item in enumerate(golden_dataset):
        q = item["question"]
        sport = item.get("file_name")
        print(f"[{idx+1}/{len(golden_dataset)}] Retrieving for '{item.get('question_id')}' ({sport})...")
        
        # 1. Normal ParentDocumentRetriever
        item["normal_retriever_context"] = pipeline.retriever.invoke(q)
        
        # 2. Advanced Hybrid Parent-Child Retriever (with metadata sport filter + CrossEncoder rerank)
        item["advanced_retriever_context"] = pipeline.advanced_parent_child_retrieval(
            query=q,
            top_k_children=DEFAULT_TOP_K_CHILDREN,
            top_n_rerank=DEFAULT_TOP_N_RERANK,
            sport=sport
        )
    print("Live retrieval complete.\n")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_evaluation(
    dataset_path: str = DEFAULT_DATASET_PKL,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    re_retrieve: bool = False,
    k_values: Tuple[int, ...] = (1, 3, 5)
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset file '{dataset_path}' not found.")

    print(f"Loading dataset from '{dataset_path}'...")
    with open(dataset_path, "rb") as f:
        golden_dataset = pickle.load(f)

    print(f"Loaded {len(golden_dataset)} evaluation questions.")

    # Optionally re-run live retrieval
    if re_retrieve:
        run_live_retrieval(golden_dataset)

    # Evaluate Normal and Advanced Retrievers
    print("Evaluating Normal Retriever...")
    normal_df, normal_avg = evaluate_retriever(golden_dataset, "normal_retriever_context", k_values)

    print("Evaluating Advanced Retriever...")
    advanced_df, advanced_avg = evaluate_retriever(golden_dataset, "advanced_retriever_context", k_values)

    # Build Comparison Summary Table
    summary_df = pd.DataFrame({
        "Normal Retriever": normal_avg,
        "Advanced Retriever": advanced_avg,
    }).round(4)

    # Calculate absolute delta & percentage improvement
    summary_df["Improvement"] = (summary_df["Advanced Retriever"] - summary_df["Normal Retriever"]).round(4)
    summary_df["% Gain"] = (
        ((summary_df["Advanced Retriever"] - summary_df["Normal Retriever"]) / summary_df["Normal Retriever"]) * 100
    ).round(2).astype(str) + "%"

    # Save to requested CSV
    summary_df.to_csv(output_csv)
    print(f"\nSummary results saved to: '{output_csv}'")

    # Also save standard name for convenience
    if output_csv != "retriever_evaluation_summary.csv":
        summary_df.to_csv("retriever_evaluation_summary.csv")

    # Print Formatted Table
    print("\n" + "=" * 70)
    print(" RETRIEVER EVALUATION COMPARISON SUMMARY ")
    print("=" * 70)
    print(summary_df.to_string())
    print("=" * 70 + "\n")

    return summary_df, normal_df, advanced_df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Normal vs. Advanced Parent-Child Retriever on Golden Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PKL,
        help=f"Path to golden dataset pickle (default: {DEFAULT_DATASET_PKL})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})"
    )
    parser.add_argument(
        "--re-retrieve",
        action="store_true",
        help="Re-run live retrieval through current vectorstore before evaluating."
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_csv=args.output,
        re_retrieve=args.re_retrieve
    )
