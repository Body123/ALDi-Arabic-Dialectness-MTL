import pandas as pd
import torch
from torch.utils.data import Dataset


def load_tsv(path):
    """Read a NADI 2024 Subtask 2 style TSV with at least `sentence` and `ALDi` columns."""
    df = pd.read_csv(path, sep="\t")
    missing = {"sentence", "ALDi"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return df


def create_class_label(df, num_classes, strategy="cut"):
    """Bin the continuous ALDi score into `num_classes` labels.

    Class 0 is reserved for MSA (ALDi == 0); the dialectal scores (ALDi > 0) are
    split into `num_classes - 1` equal-width ('cut') or quantile ('qcut') bins.
    The labels drive both the stratified folds and the auxiliary classification head.
    """
    msa_df = df[df["ALDi"] == 0].copy()
    msa_df["class_label"] = 0
    dialect_df = df[df["ALDi"] > 0].copy()

    if strategy == "qcut":
        dialect_df["class_label"] = pd.qcut(
            dialect_df["ALDi"], q=num_classes - 1, labels=False, duplicates="drop"
        ) + 1
    else:
        dialect_df["class_label"] = pd.cut(
            dialect_df["ALDi"], bins=num_classes - 1, labels=False, include_lowest=True
        ) + 1

    final_df = pd.concat([msa_df, dialect_df]).sort_index()
    final_df["class_label"] = final_df["class_label"].astype(int)
    return final_df


class DialectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = dataframe.sentence.values
        self.reg_labels = dataframe.ALDi.values
        self.class_labels = dataframe.class_label.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "regression_labels": torch.tensor(self.reg_labels[idx], dtype=torch.float),
            "class_labels": torch.tensor(self.class_labels[idx], dtype=torch.long),
        }
