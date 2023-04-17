import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model import Block


@dataclass
class A2CGPTEncoderConfig:
    # We use points on SO(5) as the latent space
    actor_latent_dim: int = 10
    actor_exponent_dim: int = 5
    # UTF-8
    input_vocab_size: int = 256
    # Number of layers in the encoder
    n_common_layers: int = 6
    n_critic_layers: int = 1
    n_advantage_layers: int = 1
    # Number of heads in the encoder
    n_head: int = 8
    # Transformer embedding dimension
    n_embd: int = 256
    # Dropout probability
    dropout: float = 0.1
    # Log2 of maximum block size
    log2_max_block_size: int = 5


@dataclass
class A2CGPTEncoderAction:
    shift: bool
    sampled_action: torch.Tensor


class A2CGPTEncoderModel(nn.Module):
    def __init__(self, config: A2CGPTEncoderConfig):
        super().__init__()
        self.config = config

        assert (
            self.config.actor_exponent_dim * (self.config.actor_exponent_dim - 1) // 2
            == self.config.actor_latent_dim
        )

        self.transformer = nn.ModuleDict(
            dict(
                input_embedding=nn.Embedding(
                    # + position (4 * log2_size), already_encoded, vacancy, length_target, length_importance, replay_bit
                    config.input_vocab_size + config.log2_max_block_size * 4 + 5,
                    config.n_embd,
                ),
                output_embedding=nn.Linear(
                    # Source action
                    config.actor_latent_dim
                    # Exponented action in SO(5)
                    + self.config.actor_exponent_dim * self.config.actor_exponent_dim
                    # + position (8 * log2_size)
                    + config.log2_max_block_size * 4
                    # + already_encoded, vacancy, replay_bit
                    + 3,
                    config.input_vocab_size,
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(config)
                        for _ in range(
                            config.n_common_layers
                            + config.n_critic_layers
                            + config.n_advantage_layers
                        )
                    ]
                ),
                actor_head=nn.Linear(
                    config.n_embd, 2 + config.actor_latent_dim * 2 + 1
                ),  # 2 Outputs for pre-actor critic, mean, logvar, and shift bit
                critic_merger=nn.Linear(
                    config.n_embd + config.actor_latent_dim * 2 + 1, config.n_embd
                ),
                critic_head=nn.Linear(config.n_embd, 2),  # mean and logvar
                advantage_merger=nn.Linear(
                    config.n_embd
                    + config.actor_latent_dim
                    + config.actor_latent_dim * config.actor_latent_dim
                    + 1,  # +1 for shift operation
                    config.n_embd,
                ),
                advantage_head=nn.Linear(config.n_embd, 2),  # mean and logvar
            )
        )
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self):
        torch.nn.init.normal_(
            self.weight,
            mean=0.0,
            std=0.02
            / math.sqrt(
                self.config.n_common_layers
                + self.config.n_critic_layers
                + self.config.n_advantage_layers
            ),
        )

    def sample_action(
        self,
        inputs: torch.Tensor,  # Tensor of ints
        outputs_shifts: torch.Tensor,  # Tensor of bools
        output_actions: torch.Tensor,  # Tensor of sampled actions, or zeros on shifts
    ):
        pass
