"""Train the multi-task ALDi model with stratified k-fold cross-validation.

  python train.py --config configs/final_system.json --train_file data/train.tsv
  python train.py --config configs/final_system.json --train_file data/train.tsv --folds 2
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, Trainer, TrainingArguments

from aldi_mtl.data import DialectDataset, create_class_label, load_tsv
from aldi_mtl.model import BertForMultiTask
from aldi_mtl.plots import plot_distributions, plot_fold_visuals, plot_summary_visuals
from aldi_mtl.training import DynamicAlphaCallback, build_model_config, compute_metrics, log_fold_performance

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task (regression + auxiliary classification) ALDi training.")
    p.add_argument("--config", default="configs/final_system.json", help="JSON file with hyperparameters.")
    p.add_argument("--train_file", required=True, help="TSV with `sentence` and `ALDi` columns.")
    p.add_argument("--output_dir", default=None, help="Default: runs/<config name>.")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Only train these folds (1-based), e.g. --folds 2. Default: all folds.")
    p.add_argument("--train_full", action="store_true",
                   help="After cross-validation, also train one model on all the training data.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    output_dir = args.output_dir or os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Outputs will be saved to %s", output_dir)

    df = create_class_label(load_tsv(args.train_file), cfg["num_classes"], strategy=cfg["binning_strategy"])
    logger.info("Class label distribution:\n%s", df["class_label"].value_counts().sort_index())
    plot_distributions(df, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    class_weights = compute_class_weight("balanced", classes=np.unique(df["class_label"]), y=df["class_label"])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = build_model_config(cfg)
    class_names = ["MSA"] + [f"Bin {i}" for i in range(1, cfg["num_classes"])]

    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["random_state"])
    all_metrics, all_histories, all_predictions, best_checkpoints = [], [], [], {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["class_label"]), start=1):
        if args.folds and fold not in args.folds:
            continue
        logger.info("========== Fold %d/%d ==========", fold, cfg["n_splits"])
        train_df, dev_df = df.iloc[train_idx], df.iloc[val_idx]
        train_dataset = DialectDataset(train_df, tokenizer, cfg["max_len"])
        dev_dataset = DialectDataset(dev_df, tokenizer, cfg["max_len"])

        model = BertForMultiTask.from_pretrained(
            cfg["model_name"], config=model_config, class_weights=class_weights.to(device)
        )
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"fold_{fold}"),
            num_train_epochs=cfg["epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"],
            max_grad_norm=cfg["max_grad_norm"],
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            save_total_limit=cfg["save_total_limit"],
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            label_names=["regression_labels", "class_labels"],
            seed=cfg["random_state"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[DynamicAlphaCallback()],
        )
        trainer.train()

        metrics = trainer.evaluate()
        log_fold_performance(fold, metrics, os.path.join(output_dir, "cv_results.txt"))
        best_checkpoints[fold] = trainer.state.best_model_checkpoint
        logger.info("Fold %d: %s | best checkpoint: %s", fold, metrics, best_checkpoints[fold])

        predictions = trainer.predict(dev_dataset)
        plot_fold_visuals(predictions.label_ids, predictions.predictions[0], predictions.predictions[1],
                          fold, output_dir, class_names)
        all_metrics.append(metrics)
        all_histories.append(trainer.state.log_history)
        all_predictions.append({"true_reg": predictions.label_ids[0], "pred_reg": predictions.predictions[0]})

    plot_summary_visuals(all_histories, all_metrics, all_predictions, output_dir)
    rmses = [m["eval_rmse"] for m in all_metrics]
    f1s = [m["eval_f1_weighted"] for m in all_metrics]
    summary = (
        f"Cross-validation over {len(all_metrics)} fold(s)\n"
        f"Average RMSE: {np.mean(rmses):.4f} (±{np.std(rmses):.4f})\n"
        f"Average auxiliary-head weighted F1: {np.mean(f1s):.4f} (±{np.std(f1s):.4f})\n"
        "Best checkpoint per fold (lowest validation RMSE):\n"
        + "".join(f"  fold {k}: {v}\n" for k, v in best_checkpoints.items())
    )
    print(summary)
    with open(os.path.join(output_dir, "final_summary.txt"), "w") as f:
        f.write("--- CONFIGURATION ---\n" + json.dumps(cfg, indent=4) + "\n\n--- RESULTS ---\n" + summary)

    if args.train_full:
        logger.info("Training a model on the entire training set")
        full_model = BertForMultiTask.from_pretrained(
            cfg["model_name"], config=model_config, class_weights=class_weights.to(device)
        )
        full_dir = os.path.join(output_dir, "full_model")
        full_args = TrainingArguments(
            output_dir=full_dir,
            num_train_epochs=max(1, int(cfg["epochs"] * 0.8)),
            per_device_train_batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            warmup_steps=cfg["warmup_steps"],
            weight_decay=cfg["weight_decay"],
            evaluation_strategy="no",
            save_strategy="no",
            seed=cfg["random_state"],
        )
        full_trainer = Trainer(model=full_model, args=full_args,
                               train_dataset=DialectDataset(df, tokenizer, cfg["max_len"]), tokenizer=tokenizer)
        full_trainer.train()
        full_trainer.save_model(full_dir)
        tokenizer.save_pretrained(full_dir)
        logger.info("Full-data model saved to %s", full_dir)


if __name__ == "__main__":
    main()
