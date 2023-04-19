import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from a2c.types import GPTAutoTokenizerRecorderSession


class GPTAutoTokenizerEncoderReplayDataset(Dataset):
    """A class for the dataloader from replays"""

    def __init__(self, replays: list[GPTAutoTokenizerRecorderSession]):
        self.replays = replays
        self.encoder_indices = [
            (replay_idx, encoder_shift)
            for replay_idx, replay in enumerate(replays)
            for encoder_shift in range(replay.input_length + replay.latent_length)
        ]
        random.shuffle(self.encoder_indices)

    def __len__(self):
        return len(self.encoder_indices)

    def __getitem__(self, idx):
        raise NotImplementedError("This is not implemented yet")


class GPTAutoTokenizerDecoderReplayDataset(Dataset):
    """A class for the dataloader from replays"""

    def __init__(self, replays: list[GPTAutoTokenizerRecorderSession]):
        self.replays = replays
        self.decoder_indices = [
            (replay_idx, decoder_shift)
            for replay_idx, replay in enumerate(replays)
            for decoder_shift in range(
                replay.latent_length + replay.reconstructed_length
            )
        ]
        random.shuffle(self.decoder_indices)

    def __len__(self):
        return len(self.decoder_indices)

    def __getitem__(self, idx):
        raise NotImplementedError("This is not implemented yet")
