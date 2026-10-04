import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from utils import DEFAULT_EMBEDDING_MODEL

DEFAULT_OUTPUT_DIR = "ragas_evaluations"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.8-27b:free"


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline using Ragas (Ollama or OpenRouter)")
    parser.add_argument("--eval-id", type=str, default="eval_1", help="Evaluation identifier (e.g. eval_1)")
    parser.add_argument("--provider", type=str, choices=["ollama", "openrouter"], default="ollama", help="LLM provider: 'ollama' (local/Colab) or 'openrouter' (default: ollama)")
    parser.add_argument("--dataset", type=str, default="ragas_dataset.json", help="Path to prepared ragas dataset")
    parser.add_argument("--start", type=int, default=0, help="Start question index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End question index (default: last)")
    parser.add_argument("--judge-model", type=str, default=None, help="Model name (default: 'qwen3:8b' for ollama, 'qwen/qwen3.8-27b:free' for openrouter)")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434", help="Ollama base URL (for local or Colab tunnel)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)")
    parser.add_argument("--timeout", type=int, default=300, help="Evaluation timeout in seconds (default: 300)")
    parser.add_argument("--strictness", type=int, default=1, help="Answer relevancy strictness (default: 1)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file '{args.dataset}' not found.")

    with open(args.dataset, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    valid_entries = [
        item for item in raw_data
        if item.get("answer") and item.get("contexts")
    ]

    total = len(valid_entries)
    if total == 0:
        print(f"No generated answers found in '{args.dataset}'. Please run prepare_ragas_dataset.py first.")
        return

    start = max(0, min(args.start, total - 1))
    end = total - 1 if args.end is None else max(0, min(args.end, total - 1))
    if start > end:
        start, end = end, start

    selected_entries = valid_entries[start : end + 1]
    print(f"Evaluating {len(selected_entries)} questions (indices {start} to {end}) from '{args.dataset}'...")

    dataset_dict = {
        "question_id": [item.get("question_id", f"q_{i}") for i, item in enumerate(selected_entries, start=start)],
        "question": [item["question"] for item in selected_entries],
        "answer": [item["answer"] for item in selected_entries],
        "contexts": [item["contexts"] for item in selected_entries],
        "ground_truth": [item.get("ground_truth", "") for item in selected_entries],
    }

    eval_dataset = Dataset.from_dict(dataset_dict)

    judge_model = args.judge_model
    if not judge_model:
        judge_model = DEFAULT_OPENROUTER_MODEL if args.provider == "openrouter" else DEFAULT_OLLAMA_MODEL

    if args.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment or .env file.")
        base_llm = ChatOpenAI(
            model=judge_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=args.temperature,
            default_headers={"HTTP-Referer": "https://localhost", "X-Title": "RAG Evaluation"},
            max_retries=5,
        )
    else:
        base_llm = ChatOllama(
            model=judge_model,
            base_url=args.ollama_base_url,
            temperature=args.temperature,
        )

    judge_llm = LangchainLLMWrapper(base_llm)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    )

    answer_relevancy.strictness = args.strictness

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    run_config = RunConfig(max_workers=1, timeout=args.timeout, max_retries=10, max_wait=60)

    print(f"\n--- Running Ragas Evaluation with {args.provider.title()} ({judge_model}) ---")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    new_df = result.to_pandas()

    new_df["question_id"] = [
        item.get("question_id", f"q_{i}")
        for i, item in enumerate(selected_entries, start=start)
    ]
    new_df["question"] = [item["question"] for item in selected_entries]
    new_df["ground_truth"] = [item.get("ground_truth", "") for item in selected_entries]
    new_df["answer"] = [item["answer"] for item in selected_entries]

    metric_names = [m.name for m in metrics]
    core_cols = ["question_id", "question", "ground_truth", "answer"] + [
        c for c in metric_names if c in new_df.columns
    ]
    remaining_cols = [c for c in new_df.columns if c not in core_cols]
    final_cols = [c for c in core_cols if c in new_df.columns] + remaining_cols
    new_df = new_df[final_cols]


    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"ragas_evaluation_results_{args.eval_id}.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_df = existing_df[existing_df["question_id"] != "MEAN_AVERAGE"].copy()

        new_ids = set(new_df["question_id"].astype(str))
        preserved_df = existing_df[~existing_df["question_id"].astype(str).isin(new_ids)]
        combined_df = pd.concat([preserved_df, new_df], ignore_index=True)
    else:
        combined_df = new_df.copy()

    dataset_id_order = {item.get("question_id", f"q_{i}"): i for i, item in enumerate(valid_entries)}
    combined_df["_sort_order"] = combined_df["question_id"].astype(str).map(lambda x: dataset_id_order.get(x, 999999))
    combined_df.sort_values("_sort_order", inplace=True)
    combined_df.drop(columns=["_sort_order"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    eval_metric_cols = [m for m in metric_names if m in combined_df.columns]
    summary_series = combined_df[eval_metric_cols].apply(pd.to_numeric, errors="coerce").mean(skipna=True)

    summary_row = {col: "" for col in combined_df.columns}
    summary_row["question_id"] = "MEAN_AVERAGE"
    for m in eval_metric_cols:
        if m in summary_series and not pd.isna(summary_series[m]):
            summary_row[m] = round(float(summary_series[m]), 4)

    df_with_summary = pd.concat([combined_df, pd.DataFrame([summary_row])], ignore_index=True)
    df_with_summary.to_csv(output_path, index=False, encoding="utf-8")

    print("\n=======================================================")
    print(" RAGAS EVALUATION COMPLETED")
    print("=======================================================")
    print(f"Batch evaluated : {len(selected_entries)} questions (indices {start} to {end})")
    print(f"Total in results: {len(combined_df)} questions")
    print(f"Results saved to: {output_path}")
    print("\n--- Summary Metric Scores (Mean of all evaluated) ---")
    for m in eval_metric_cols:
        if m in summary_series and not pd.isna(summary_series[m]):
            val = float(summary_series[m])
            print(f" - {m:<20}: {val:.4f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
