import os
import json
import argparse

INPUT_FILE = "golden_dataset.json"
OUTPUT_FILE = "ragas_dataset.json"


def save_dataset(data, filepath=OUTPUT_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for RAGAS evaluation")
    parser.add_argument("--start", type=int, default=0, help="Start question index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End question index (default: last)")
    args = parser.parse_args()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    total = len(golden_data)
    if total == 0:
        print("No questions found in", INPUT_FILE)
        return

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            ragas_data = json.load(f)
    else:
        ragas_data = [
            {
                "question_id": item.get("question_id", f"q_{i}"),
                "question": item["question"],
                "ground_truth": item.get("ground_truth", ""),
                "answer": "",
                "contexts": []
            }
            for i, item in enumerate(golden_data)
        ]

    while len(ragas_data) < total:
        i = len(ragas_data)
        ragas_data.append({
            "question_id": golden_data[i].get("question_id", f"q_{i}"),
            "question": golden_data[i]["question"],
            "ground_truth": golden_data[i].get("ground_truth", ""),
            "answer": "",
            "contexts": []
        })

    start = max(0, min(args.start, total - 1))
    end = total - 1 if args.end is None else max(0, min(args.end, total - 1))
    if start > end:
        start, end = end, start

    print(f"Processing questions {start} to {end} (Total questions: {total})...")

    from rag_pipeline import get_pipeline
    pipeline = get_pipeline()

    for idx in range(start, end + 1):
        item = golden_data[idx]
        q_id = item.get("question_id", f"q_{idx}")
        print(f"[{idx + 1}/{total}] Generating for '{q_id}'...")

        try:
            res = pipeline.query_sync(item["question"], return_sources=True)
            answer = res["answer"] if isinstance(res, dict) else res.answer
            contexts = [doc.page_content for doc in res.get("sources", [])]

            ragas_data[idx]["question_id"] = q_id
            ragas_data[idx]["question"] = item["question"]
            ragas_data[idx]["ground_truth"] = item.get("ground_truth", "")
            ragas_data[idx]["answer"] = answer
            ragas_data[idx]["contexts"] = contexts

            save_dataset(ragas_data)

        except Exception as e:
            print(f"Error on question index {idx} ({q_id}): {e}")
            print(f"Progress up to question {idx - 1} has been saved.")
            break

    print(f"Done. Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
