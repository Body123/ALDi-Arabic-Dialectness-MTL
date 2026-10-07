import logging

import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2020) for the auxiliary classification head."""

    def __init__(self, alpha=1, gamma=2, reduction="mean", weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BertForMultiTask(PreTrainedModel):
    """Shared encoder whose [CLS] vector feeds a regression head (ALDi score) and a
    classification head (ALDi bins). Training loss: alpha * L_reg + (1 - alpha) * L_cls.
    Only the regression output is used at inference.
    """

    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.config = config
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(config.num_labels))

        self.bert = AutoModel.from_config(config)

        def create_head(input_size, output_size, hidden_layers, dropout_rate):
            layers = []
            last_size = input_size
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(last_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                layers.append(nn.PReLU(init=self.config.prelu_init))
                layers.append(nn.Dropout(dropout_rate))
                last_size = hidden_size
            layers.append(nn.Linear(last_size, output_size))
            return nn.Sequential(*layers)

        self.regressor = create_head(config.hidden_size, 1, config.head_hidden_layers, config.head_dropout)
        self.classifier = create_head(
            config.hidden_size, config.num_labels, config.head_hidden_layers, config.head_dropout
        )
        self.post_init()

    def forward(self, input_ids, attention_mask, regression_labels=None, class_labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]

        reg_logits = self.regressor(pooled_output).squeeze(-1)
        if torch.isnan(reg_logits).any():
            logger.warning("NaN detected in regression output; clamping values")
        reg_logits = torch.nan_to_num(reg_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        reg_logits = torch.clamp(reg_logits, -10, 10)

        class_logits = self.classifier(pooled_output)
        class_logits = torch.clamp(class_logits, -50, 50)

        loss = None
        if regression_labels is not None and class_labels is not None:
            if torch.isnan(regression_labels).any() or torch.isinf(regression_labels).any():
                logger.warning("NaN/Inf in regression labels; replacing with 0")
                regression_labels = torch.nan_to_num(regression_labels, nan=0.0, posinf=0.0, neginf=0.0)

            if self.config.regression_loss_type == "huber":
                reg_loss = nn.HuberLoss()(reg_logits, regression_labels)
            else:
                reg_loss = nn.MSELoss()(reg_logits, regression_labels)
            if torch.isnan(reg_loss):
                logger.warning("NaN in regression loss; setting to 0")
                reg_loss = torch.tensor(0.0, device=reg_loss.device)

            if self.config.loss_type == "focal":
                loss_fn = FocalLoss(gamma=self.config.focal_loss_gamma, weight=self.class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            class_loss = loss_fn(class_logits.view(-1, self.config.num_labels), class_labels.view(-1))
            if torch.isnan(class_loss):
                logger.warning("NaN in classification loss; setting to 0")
                class_loss = torch.tensor(0.0, device=class_loss.device)

            loss = self.config.alpha * reg_loss + (1 - self.config.alpha) * class_loss

        return (loss, reg_logits, class_logits)
