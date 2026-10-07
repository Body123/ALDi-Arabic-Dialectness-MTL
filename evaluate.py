"""Evaluate a checkpoint on a labeled TSV (e.g. the NADI 2024 Subtask 2 dev set).

  python evaluate.py --data_file data/dev.tsv                       # released model from the Hub
  python evaluate.py --data_file data/dev.tsv --model runs/final_system/fold_2/checkpoint-9750
"""
import argparse

import numpy as np
import pandas as pd

from aldi_mtl.data import load_tsv
from aldi_mtl.inference import load_model, score


def main():
    p = argparse.ArgumentParser(description="RMSE of the ALDi regression head on a labeled TSV.")
    p.add_argument("--data_file", required=True, help="TSV with `sentence` and `ALDi` columns.")
    p.add_argument("--model", default="", help="Local checkpoint folder. Default: the released model on the Hub.")
    p.add_argument("--output", default="predictions.csv", help="Where to write per-sentence predictions.")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    df = load_tsv(args.data_file)
    tokenizer, model, device = load_model(args.model)
    print(f"Scoring {len(df)} sentences on {device.upper()} ...")
    preds = np.array(score(list(df["sentence"]), tokenizer, model, device, args.batch_size, clip=False))
    gold = df["ALDi"].to_numpy(dtype=float)

    rmse = np.sqrt(np.mean((preds - gold) ** 2))
    print(f"RMSE: {rmse:.5f}")

    pd.DataFrame({
        "sentence": df["sentence"],
        "true_aldi": gold,
        "pred_aldi": preds.round(5),
        "absolute_error": np.abs(preds - gold).round(5),
    }).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
