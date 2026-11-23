import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import SEED


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
