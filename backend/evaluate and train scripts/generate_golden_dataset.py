import os
import json
import pickle
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from utils import chunking_embedding, DEFAULT_TOP_K_CHILDREN, DEFAULT_TOP_N_RERANK
from rag_pipeline import get_pipeline

DEFAULT_INPUT_JSON = "golden_dataset.json"
DEFAULT_OUTPUT_PKL = "golden_dataset.pkl"
DEFAULT_OUTPUT_JSON = "golden_dataset.json"
DEFAULT_JUDGE_MODEL = "gemma2:2b"

JUDGE_TEMPLATE = """You are an expert Information Retrieval evaluator.
Your task is to determine if the retrieved context is relevant to the question and contains the correct answer as defined by the ground truth.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context}

Evaluate whether the Retrieved Context provides the necessary information to answer the Question accurately. 

CRITICAL RULE: The Retrieved Context MUST contain the specific facts, details, or information present in the Ground Truth. If the context is only generally related to the question but is missing the actual answer provided in the Ground Truth, you must evaluate it as not relevant.

Respond ONLY with the exact phrase "relevant" if the context explicitly supports the Ground Truth answer, or "not relevant" if it is missing the necessary information. Do not include any reasoning, explanation, or punctuation."""


# ---------------------------------------------------------------------------
# Judge LLM Chain (Ollama gemma2:2b)
# ---------------------------------------------------------------------------
def create_judge_chain(model_name: str = DEFAULT_JUDGE_MODEL):
    """
    Creates an LLM judge chain using ChatOllama to determine if a context supports the ground truth.
    """
    judge_llm = ChatOllama(
        model=model_name,
        temperature=0
    )

    judge_prompt = PromptTemplate(
        input_variables=["question", "ground_truth", "context"],
        template=JUDGE_TEMPLATE
    )
    return judge_prompt | judge_llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Step 1: Retrieval (Populate Normal & Advanced Contexts)
# ---------------------------------------------------------------------------
def populate_retrieved_contexts(golden_dataset: List[Dict[str, Any]]):
    """
    Executes Normal Retriever and Advanced Parent-Child Retriever for every question.
    """
    print("\n--- Step 1: Running Normal & Advanced Retrievers ---")
    pipeline = get_pipeline()

    for idx, item in enumerate(golden_dataset):
        q = item["question"]
        sport = item.get("file_name")
        qid = item.get("question_id", f"q_{idx+1}")
        print(f"[{idx+1}/{len(golden_dataset)}] Retrieving for {qid} ({sport})...")

        # 1. Normal ParentDocumentRetriever
        normal_docs = pipeline.retriever.invoke(q)
        item["normal_retriever_context"] = normal_docs

        # 2. Advanced Retriever (Sport filter + BM25 + CrossEncoder + Parent Doc Resolution)
        advanced_docs = pipeline.advanced_parent_child_retrieval(
            query=q,
            top_k_children=DEFAULT_TOP_K_CHILDREN,
            top_n_rerank=DEFAULT_TOP_N_RERANK,
            sport=sport
        )
        item["advanced_retriever_context"] = advanced_docs

    print("Retrieval step completed for all questions.")


# ---------------------------------------------------------------------------
# Step 2: Global Ground-Truth Discovery (LLM Judge over Entire DB)
# ---------------------------------------------------------------------------
def find_globally_relevant_docs(
    golden_dataset: List[Dict[str, Any]],
    model_name: str = DEFAULT_JUDGE_MODEL
):
    """
    Scans every parent chunk in the database matching the target sport,
    and asks the Ollama Judge whether it contains the factual Ground Truth.
    Saves matching doc_ids into item['relevant_docs'].
    """
    print(f"\n--- Step 2: Evaluating Entire DB with Ollama Judge ({model_name}) ---")
    judge = create_judge_chain(model_name=model_name)

    # Load all parent documents from DocStore
    _, _, store = chunking_embedding(is_for_embedding=False)
    all_parent_ids = list(store.yield_keys())
    all_parent_docs = store.mget(all_parent_ids)
    print(f"Loaded {len(all_parent_docs)} parent documents from DocStore.")

    for i, item in enumerate(golden_dataset):
        target_file_name = item.get("file_name")
        question = item["question"]
        ground_truth = item["ground_truth"]
        qid = item.get("question_id", f"q_{i+1}")

        print(f"[{i+1}/{len(golden_dataset)}] Finding ground-truth docs for {qid} ({target_file_name})...")
        item["relevant_docs"] = []

        for doc in all_parent_docs:
            if doc is None:
                continue

            chunk_file_name = doc.metadata.get("file_name")

            # Skip documents from other sports to save compute
            if target_file_name and chunk_file_name and chunk_file_name.lower() != target_file_name.lower():
                continue

            try:
                raw_output = judge.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "context": doc.page_content
                })
                verdict = raw_output.strip().lower()

                if "relevant" in verdict and "not relevant" not in verdict:
                    doc_id = doc.metadata.get("doc_id")
                    if doc_id:
                        item["relevant_docs"].append(doc_id)
            except Exception as e:
                print(f"   Warning evaluating doc {doc.metadata.get('doc_id')}: {e}")

        print(f"   -> Found {len(item['relevant_docs'])} verified relevant parent docs.")

    print("\nGlobal ground-truth discovery complete.")


# ---------------------------------------------------------------------------
# Step 3: Label Relevance Flags
# ---------------------------------------------------------------------------
def label_retrieved_relevance(golden_dataset: List[Dict[str, Any]]):
    """
    Tags each retrieved document in normal_retriever_context and advanced_retriever_context
    with 'relevant' or 'not relevant' based on item['relevant_docs'].
    """
    print("\n--- Step 3: Labeling Retrieved Relevance ---")
    for item in golden_dataset:
        relevant_set = set(item.get("relevant_docs", []))

        item["normal_relevance"] = [
            "relevant" if (getattr(d, "metadata", {}).get("doc_id") in relevant_set) else "not relevant"
            for d in item.get("normal_retriever_context", [])
        ]

        item["advanced_relevance"] = [
            "relevant" if (getattr(d, "metadata", {}).get("doc_id") in relevant_set) else "not relevant"
            for d in item.get("advanced_retriever_context", [])
        ]
    print("Labeling complete.")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def generate_dataset(
    input_json: str = DEFAULT_INPUT_JSON,
    output_pkl: str = DEFAULT_OUTPUT_PKL,
    output_json: Optional[str] = DEFAULT_OUTPUT_JSON,
    find_relevant: bool = False,
    judge_model: str = DEFAULT_JUDGE_MODEL
):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input questions file '{input_json}' not found.")

    print(f"Loading base questions from '{input_json}'...")
    with open(input_json, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    print(f"Loaded {len(golden_dataset)} questions.")

    # Check if we should reuse existing relevant_docs from existing pkl
    if not find_relevant and os.path.exists(output_pkl):
        print(f"Preserving existing ground-truth 'relevant_docs' from '{output_pkl}' (use --find-relevant to re-scan DB with Ollama).")
        with open(output_pkl, "rb") as f:
            existing_data = pickle.load(f)
        rel_map = {item.get("question_id"): item.get("relevant_docs", []) for item in existing_data}
        for item in golden_dataset:
            qid = item.get("question_id")
            if qid in rel_map:
                item["relevant_docs"] = rel_map[qid]
    elif find_relevant:
        find_globally_relevant_docs(golden_dataset, model_name=judge_model)
    else:
        print("Note: No existing relevant_docs found and --find-relevant not set. Initializing empty relevant_docs.")
        for item in golden_dataset:
            item.setdefault("relevant_docs", [])

    # 1. Populate current retriever contexts
    populate_retrieved_contexts(golden_dataset)

    # 2. Label relevance
    label_retrieved_relevance(golden_dataset)

    # 3. Save to Pickle
    with open(output_pkl, "wb") as f:
        pickle.dump(golden_dataset, f)
    print(f"\nSaved full evaluation dataset to '{output_pkl}'.")

    # 4. Save JSON version (clean representation for text editors)
    if output_json:
        clean_json_data = []
        for item in golden_dataset:
            clean_item = {
                "question_id": item.get("question_id"),
                "file_name": item.get("file_name"),
                "question": item.get("question"),
                "ground_truth": item.get("ground_truth"),
                "relevant_docs": item.get("relevant_docs", []),
                "normal_relevance": item.get("normal_relevance", []),
                "advanced_relevance": item.get("advanced_relevance", []),
                "num_normal_retrieved": len(item.get("normal_retriever_context", [])),
                "num_advanced_retrieved": len(item.get("advanced_retriever_context", []))
            }
            clean_json_data.append(clean_item)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(clean_json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved human-readable summary to '{output_json}'.")

    print("\nGolden dataset generation complete! You can now run 'python retriever_evaluation.py'.")
    return golden_dataset


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate, enrich, and evaluate the Golden Dataset for RAG benchmarking"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=f"Input JSON containing base Q&A pairs (default: {DEFAULT_INPUT_JSON})"
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=DEFAULT_OUTPUT_PKL,
        help=f"Output PKL path (default: {DEFAULT_OUTPUT_PKL})"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})"
    )
    parser.add_argument(
        "--find-relevant",
        action="store_true",
        help="Run Ollama Judge across entire DocStore to discover ground-truth relevant_docs (slow, use for new questions)."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Ollama model name for judge (default: {DEFAULT_JUDGE_MODEL})"
    )

    args = parser.parse_args()

    generate_dataset(
        input_json=args.input,
        output_pkl=args.output_pkl,
        output_json=args.output_json,
        find_relevant=args.find_relevant,
        judge_model=args.judge_model
    )
