import logging
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from transformers import AutoConfig, TrainerCallback

logger = logging.getLogger(__name__)


def build_model_config(cfg):
    """Base encoder config extended with the multi-task settings the model reads."""
    model_config = AutoConfig.from_pretrained(cfg["model_name"])
    model_config.num_labels = cfg["num_classes"]
    model_config.loss_type = cfg["loss_type"]
    model_config.focal_loss_gamma = cfg["focal_loss_gamma"]
    model_config.regression_loss_type = cfg["regression_loss_type"]
    model_config.prelu_init = cfg["prelu_init"]
    model_config.head_dropout = cfg["head_dropout"]
    model_config.head_hidden_layers = cfg["head_hidden_layers"]
    model_config.use_dynamic_alpha = cfg["use_dynamic_alpha"]
    model_config.dynamic_alpha_switch_epoch = cfg["dynamic_alpha_switch_epoch"]
    model_config.alpha_initial = cfg["alpha_initial"]
    model_config.alpha_final = cfg["alpha_final"]
    model_config.alpha = cfg["alpha_initial"] if cfg["use_dynamic_alpha"] else cfg["alpha"]
    return model_config


def compute_metrics(p):
    """RMSE of the regression head (model selection metric) plus F1 of the auxiliary head."""
    reg_preds, class_preds_logits = p.predictions
    reg_labels, class_labels = p.label_ids

    if np.isnan(reg_preds).any():
        logger.warning("NaN in regression predictions; replacing with 0 for metric calculation")
        reg_preds = np.nan_to_num(reg_preds, nan=0.0)

    rmse = np.sqrt(mean_squared_error(reg_labels, reg_preds))
    class_preds = np.argmax(class_preds_logits, axis=1)
    return {
        "rmse": rmse if not np.isnan(rmse) else 10.0,
        "f1_weighted": f1_score(class_labels, class_preds, average="weighted"),
        "f1_micro": f1_score(class_labels, class_preds, average="micro"),
    }


class DynamicAlphaCallback(TrainerCallback):
    """Optionally switch the loss weight from `alpha_initial` to `alpha_final` at a given epoch."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not model.config.use_dynamic_alpha:
            return
        config = model.config
        epoch = int(state.epoch)
        new_alpha = config.alpha_initial if epoch < config.dynamic_alpha_switch_epoch else config.alpha_final
        if config.alpha != new_alpha:
            logger.info("Epoch %d: changing alpha from %.2f to %.2f", epoch, config.alpha, new_alpha)
            config.alpha = new_alpha


def log_fold_performance(fold_idx, metrics, save_path):
    with open(save_path, "a") as f:
        f.write(f"Fold {fold_idx} - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("-" * 40 + "\n")
