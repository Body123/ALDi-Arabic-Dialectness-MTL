import os

import torch
from transformers import AutoConfig, AutoTokenizer

from .model import BertForMultiTask

DEFAULT_MODEL = "asakr-ai/ALDi-Arabic-Dialectness-MTL"
MAX_LEN = 256  # sequence length used in training


def load_model(model="", device=None):
    """Load a checkpoint from a local folder, or the released model from the Hugging Face Hub."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model and os.path.isdir(model):
        source, kwargs = model, {}
    else:
        source, kwargs = (model or DEFAULT_MODEL), {"subfolder": "mtl_model"}
    config = AutoConfig.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    net = BertForMultiTask.from_pretrained(source, config=config, **kwargs).to(device).eval()
    return tokenizer, net, device


@torch.no_grad()
def score(sentences, tokenizer, model, device, batch_size=32, clip=True):
    """ALDi scores for a list of sentences (regression head only)."""
    scores = []
    for i in range(0, len(sentences), batch_size):
        enc = tokenizer(
            [str(s) for s in sentences[i : i + batch_size]],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        _, reg, _ = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        scores.extend((reg.clamp(0, 1) if clip else reg).tolist())
    return scores
