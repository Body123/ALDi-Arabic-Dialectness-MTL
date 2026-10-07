"""Predict ALDi scores (0 = MSA, 1 = fully dialectal) with the released or a local model.

  python predict.py --text "شو هالحكي يا زلمة"
  python predict.py --input sentences.csv --output predictions.csv
  python predict.py --input sentences.txt --model runs/final_system/fold_2/checkpoint-9750
"""
import argparse
import csv
import sys

from aldi_mtl.inference import load_model, score


def read_sentences(path, column):
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    delimiter = "\t" if path.endswith(".tsv") else ","
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if column not in (reader.fieldnames or []):
            sys.exit(f"Column '{column}' not found in {path}. Available: {reader.fieldnames}")
        return [row[column] for row in reader]


def main():
    p = argparse.ArgumentParser(description="Predict ALDi (Arabic Level of Dialectness) scores.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", action="append", help="Sentence to score (repeat for several).")
    src.add_argument("--input", help="A .csv/.tsv file (with a sentence column) or .txt file (one per line).")
    p.add_argument("--column", default="sentence", help="Sentence column name for .csv/.tsv (default: sentence).")
    p.add_argument("--output", help="Write results to this CSV instead of printing them.")
    p.add_argument("--model", default="", help="Local checkpoint folder. Default: the released model on the Hub.")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    sentences = args.text or read_sentences(args.input, args.column)
    tokenizer, model, device = load_model(args.model)
    print(f"Scoring {len(sentences)} sentence(s) on {device.upper()} ...", file=sys.stderr)
    scores = score(sentences, tokenizer, model, device, args.batch_size)

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "aldi_score"])
            writer.writerows(zip(sentences, (round(s, 4) for s in scores)))
        print(f"Saved {len(scores)} predictions to {args.output}", file=sys.stderr)
    else:
        for sentence, s in zip(sentences, scores):
            print(f"{s:.4f}\t{sentence}")


if __name__ == "__main__":
    main()
