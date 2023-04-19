from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from a2c.types import GPTAutoTokenizerRecorderSession


class GPTAutoTokenizerDataset(Dataset):
    """A class for the dataloader from replays"""

    def __init__(self, replays: list[GPTAutoTokenizerRecorderSession]):
        self.replays = replays
