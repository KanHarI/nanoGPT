import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from lie_theory import so_to_SO
from model import Block


@dataclass
class A2CGPTEncoderConfig:
    # We use points on SO(5) as the latent space
    # 10 points - the lie dimension of SO(5) is 10
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


def frequencies_block(log2_size: int):
    frequencies = []
    for i in range(log2_size):
        frequencies.append(2 * math.pi / 2**log2_size * (2 ** (i - 0.5)))
        frequencies.append(2 * math.pi / 2**log2_size * (2**i))
    block = []
    for i in range(2**log2_size):
        block.append([])
        for freq in frequencies:
            block[-1].append(math.sin(freq * i))
            block[-1].append(math.cos(freq * i))
    return torch.Tensor(block, requires_grad=False)


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
                wte=nn.Embedding(
                    config.input_vocab_size,
                    # +4:
                    # +1 for already_processed
                    # +1 for vacancy
                    # +1 target len ratio
                    # +1 for target len importance
                    # +1 for marking inputs
                    config.n_embd - (config.log2_max_block_size * 4 + 5),
                ),
                output_embedding=nn.Linear(
                    # Source action
                    config.actor_latent_dim
                    # Exponented action in SO(5)
                    + config.actor_exponent_dim * config.actor_exponent_dim
                    # + position (8 * log2_size)
                    + config.log2_max_block_size * 4
                    # + already_encoded, vacancy, shift, output indicator
                    + 4,
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
        self.frequencies_block = frequencies_block(config.log2_max_block_size)
        self.standard_attn_mask = torch.ones(
            config.log2_max_block_size * 2, config.log2_max_block_size * 2
        )
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

    @torch.no_grad()
    def sample_nograd(
        self,
        input: str,
        target_len_ratio: float,
        target_len_importance: float,
        max_actions: Optional[int] = None,
    ) -> list[A2CGPTEncoderAction]:
        input_bytes = input.encode("utf-8")
        shifts_to_end = len(input_bytes)
        shifted_bytes = 0
        input_embeddings = self.transformer.wte(torch.LongTensor(input_bytes))
        output_actions: list[A2CGPTEncoderAction] = []
        num_items_ahead = 2 ** (self.config.log2_max_block_size - 1)
        while shifted_bytes < shifts_to_end and len(output_actions) < max_actions:
            # Create input embeddings
            processed_visible_bytes = min(shifted_bytes, num_items_ahead)
            input_processed = input_embeddings[
                shifted_bytes - processed_visible_bytes : shifted_bytes
            ]
            input_context = input_embeddings[
                shifted_bytes : shifted_bytes + num_items_ahead
            ]
            # Pad to num_items_ahead with zeros
            full_input_context = torch.cat(
                [
                    torch.zeros(
                        num_items_ahead - processed_visible_bytes,
                        input_context.shape[1],
                    ),
                    input_processed,
                    input_context,
                    torch.zeros(
                        num_items_ahead - input_context.shape[0],
                        input_context.shape[1],
                    ),
                ]
            )
            # Add position embedding
            full_input_context = torch.stack(
                [
                    full_input_context,
                    self.frequencies_block,
                    # Already processed - by position
                    torch.cat(
                        [torch.ones(num_items_ahead), torch.zeros(num_items_ahead)]
                    ),
                    # Vacancy bit
                    torch.cat(
                        [
                            torch.zeros(num_items_ahead - processed_visible_bytes),
                            torch.ones(processed_visible_bytes),
                            torch.ones(input_processed.shape[0]),
                            torch.zeros(num_items_ahead - input_context.shape[0]),
                        ]
                    ),
                    # Target length ratio
                    torch.ones(num_items_ahead * 2) * target_len_ratio,
                    # Target length importance
                    torch.ones(num_items_ahead * 2) * target_len_importance,
                    # Inputs indicator
                    torch.zeros(num_items_ahead * 2),
                ],
                dim=1,
            )
            # Create output embeddings
            vacant_pre_outputs_count = max(num_items_ahead - len(output_actions), 0)
            relevant_actions = output_actions[-num_items_ahead:]
            output = torch.zeros(
                num_items_ahead * 2,
                self.config.actor_latent_dim
                + self.config.actor_exponent_dim * self.config.actor_exponent_dim,
            )
            output = torch.stack(
                [
                    output,
                    self.frequencies_block,
                    # processed bit
                    torch.cat(
                        [torch.ones(num_items_ahead), torch.zeros(num_items_ahead)]
                    ),
                    # Vacancy bit
                    torch.cat(
                        [
                            torch.zeros(num_items_ahead - processed_visible_bytes),
                            torch.ones(processed_visible_bytes),
                            torch.ones(input_processed.shape[0]),
                            torch.zeros(num_items_ahead - input_context.shape[0]),
                        ]
                    ),
                    # Shift bit
                    torch.zeros(num_items_ahead * 2),
                    # Output indicator
                    torch.ones(num_items_ahead * 2),
                ],
                dim=1,
            )
            ptr = num_items_ahead - len(relevant_actions)
            while ptr < num_items_ahead:
                action = relevant_actions[ptr]
                if action.shift:
                    output[ptr, -2] = 1
                else:
                    exponentiated_action = so_to_SO(action.sampled_action)
                    output[ptr, : self.config.actor_latent_dim] = action.sampled_action
                    output[
                        ptr,
                        self.config.actor_latent_dim : ptr
                        + self.config.actor_latent_dim
                        + self.config.actor_exponent_dim
                        * self.config.actor_exponent_dim,
                    ] = exponentiated_action
