import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


def plot_distributions(df, output_dir):
    """Continuous ALDi scores and the binned class labels."""
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["ALDi"], kde=True, bins=30)
    plt.title("Distribution of Continuous ALDi Scores")
    plt.xlabel("ALDi Score")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    sns.countplot(x="class_label", data=df, palette="viridis")
    plt.title("Distribution of Class Labels")
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_distributions.png"))
    plt.close()


def plot_fold_visuals(labels, preds_reg, preds_class_logits, fold, output_dir, class_names):
    """Confusion matrix of the auxiliary head and true-vs-predicted ALDi for one fold."""
    preds_class = np.argmax(preds_class_logits, axis=1)
    print(classification_report(labels[1], preds_class, labels=list(range(len(class_names))),
                                target_names=class_names, zero_division=0))

    cm = confusion_matrix(labels[1], preds_class, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_fold_{fold}.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=labels[0], y=preds_reg, alpha=0.5)
    lo, hi = min(labels[0]), max(labels[0])
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--")
    plt.title(f"Regression Performance - Fold {fold}")
    plt.xlabel("True ALDi Score")
    plt.ylabel("Predicted ALDi Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"regression_performance_fold_{fold}.png"))
    plt.close()


def plot_summary_visuals(all_histories, all_fold_metrics, all_predictions, output_dir):
    """Average validation loss, per-fold metrics, and a combined true-vs-predicted plot."""
    eval_losses = [[log["eval_loss"] for log in h if "eval_loss" in log] for h in all_histories]
    n = min(len(x) for x in eval_losses)
    if n > 0:
        eval_losses = np.array([x[:n] for x in eval_losses])
        avg, std = eval_losses.mean(axis=0), eval_losses.std(axis=0)
        steps = np.arange(1, n + 1)
        plt.figure(figsize=(12, 7))
        plt.plot(steps, avg, "r-o", label="Average Validation Loss")
        plt.fill_between(steps, avg - std, avg + std, color="red", alpha=0.2, label="Std. Dev.")
        plt.title("Average Validation Loss Across Folds")
        plt.xlabel("Evaluation")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "summary_avg_loss_curves.png"))
        plt.close()

    metrics_df = pd.DataFrame({
        "Fold": [f"Fold {i + 1}" for i in range(len(all_fold_metrics))],
        "RMSE": [m["eval_rmse"] for m in all_fold_metrics],
        "Weighted F1": [m["eval_f1_weighted"] for m in all_fold_metrics],
    })
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x="Fold", y="RMSE", data=metrics_df, palette="coolwarm")
    plt.title("Validation RMSE per Fold")
    plt.subplot(1, 2, 2)
    sns.barplot(x="Fold", y="Weighted F1", data=metrics_df, palette="viridis")
    plt.title("Auxiliary-head Weighted F1 per Fold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_metrics_per_fold.png"))
    plt.close()

    true_reg = np.concatenate([p["true_reg"] for p in all_predictions])
    pred_reg = np.concatenate([p["pred_reg"] for p in all_predictions])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=true_reg, y=pred_reg, alpha=0.3, s=15)
    plt.plot([true_reg.min(), true_reg.max()], [true_reg.min(), true_reg.max()], color="red", linestyle="--")
    plt.title("Combined Regression Performance (All Folds)")
    plt.xlabel("True ALDi Score")
    plt.ylabel("Predicted ALDi Score")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "summary_combined_regression_plot.png"))
    plt.close()
