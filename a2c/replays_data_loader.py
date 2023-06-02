import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from a2c.a2c_types import GPTAutoTokenizerConfig, GPTAutoTokenizerRecorderSession


class GPTAutoTokenizerEncoderReplayDataset(Dataset):
    """A class for the dataloader from replays"""

    def __init__(
        self,
        replays: list[GPTAutoTokenizerRecorderSession],
        config: GPTAutoTokenizerConfig,
    ):
        self.replays = replays
        self.encoder_indices = [
            (replay_idx, encoder_shift)
            for replay_idx, replay in enumerate(replays)
            for encoder_shift in range(replay.input_length + replay.latent_length)
        ]
        random.shuffle(self.encoder_indices)
        self.config = config
        self.block_size = 2**config.log2_max_block_size
        self.look_ahead = self.block_size // 2
        total_rewards_over_length = [rp.total_reward_over_length for rp in replays]
        total_rewards_over_length_with_indices = enumerate(total_rewards_over_length)
        # Reinforce good existing behavior
        self.total_rewards_over_length_with_start_indices = dict(
            sorted(
                total_rewards_over_length_with_indices,
                key=lambda x: x[1],
            )
        )
        self.total_rewards_over_length_with_end_indices = dict(
            sorted(
                total_rewards_over_length_with_indices[::-1],
                key=lambda x: x[1],
            )
        )
        self.total_reward_std = np.std(total_rewards_over_length)

    def __len__(self):
        return len(self.encoder_indices)

    def __getitem__(self, idx):
        replay = self.replays[self.encoder_indices[idx][0]]
        shift = self.encoder_indices[idx][1]
        total_reward_over_length = replay.total_reward_over_length
        reward_over_length_start_percentile = (
            self.total_rewards_over_length_with_start_indices[total_reward_over_length]
            / len(self.replays)
        )
        reward_over_length_end_percentile = (
            self.total_rewards_over_length_with_end_indices[total_reward_over_length]
            / len(self.replays)
        )
        reward_over_length_percentile = (
            reward_over_length_start_percentile + reward_over_length_end_percentile
        ) / 2
        elements_visible_before_shift = min(self.look_ahead, shift)
        elements_visible_after_shift = min(
            self.look_ahead, replay.input_length + replay.latent_length - shift
        )
        input_elements = torch.zeros(self.block_size, dtype=torch.long)
        for i in range(-elements_visible_before_shift, elements_visible_after_shift):
            input_elements[self.look_ahead + i] = replay.input_vocab[shift + i]


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
