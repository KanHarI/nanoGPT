import copy
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from a2c.lie_theory import so_to_SO
from a2c.text_loss import similar
from model import Block


@dataclass
class GPTAutoTokenizerConfig:
    # We use points on SO(n) as the latent space
    actor_exponent_dim: int = 7
    # UTF-8
    input_vocab_size: int = 256
    # Number of layers in the encoder
    n_common_layers: int = 6
    # Number of layers to use for the advantage function calculation
    n_advantage_layers: int = 2
    # Number of heads in the encoder
    n_head: int = 8
    # Transformer embedding dimension
    n_embd: int = 256
    # Dropout probability
    dropout: float = 0.1
    # Log2 of maximum block size
    log2_max_block_size: int = 5
    # Biases
    bias: bool = True
    # Loss for comparing the vocab reconstruction result
    text_loss: Callable[[Iterable, Iterable], float] = similar
    # Learning rate for the PPO A2C process
    a2c_learning_rate: float = 1e-4
    # Auxilary loss - reinforce patterns that resulted in good total reward,
    # even if missed by critic. Important early on, when critic is not
    # accurate.
    stability_relative_weight_to_reward_std: float = 5e-2
    # Reinforce the top half of the reward distribution, with bigger rewards
    # by percentiles (linearly from the min_percentile_stability_reward to 1)
    min_percentile_stability_reward: float = 0.5
    # Factor by which the KL divergence must be less than the target, otherwise
    # stop PPO training early
    early_stop_kl_divergence_target_factor: float = 2.0
    # Update the current approximated KL after ~20 batches
    kl_moving_average_iterations_constant: float = 0.9
    # Discount factor for the reward
    discount_factor: float = 0.9
    # Validation split percentile
    val_split_percentile: float = 0.2
    # Allowed critic overfit
    # If validation set loss is less than the best observed validation loss
    # multiplied by this factor, then stop PPO training early
    critic_overfit_allowed: float = 1.1


@dataclass
class VocabToLatentAction:
    shift: bool = False
    latent: Optional[torch.Tensor] = None


@dataclass
class LatentToVocabAction:
    shift: bool = False
    vocab: Optional[int] = None


@dataclass
class GPTAutoTokenizerRecorderSession:
    # L * V
    input_vocab: torch.Tensor
    # L
    encoder_extra_reward: torch.Tensor
    # L * (E * (E - 1) / 2)
    latent_representation: torch.Tensor
    # L * (E ** 2)
    exponent_representation: torch.Tensor
    # L
    decoder_reward: torch.Tensor
    # (1)
    total_reward: float
    total_reward_over_length: float


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
    result = torch.Tensor(block)
    result.requires_grad = False
    return result


class GPTAutoTokenizer(nn.Module):
    def __init__(self, config: GPTAutoTokenizerConfig):
        super().__init__()

        self.config = config
        self.latent_actor_dim = (
            config.actor_exponent_dim * (config.actor_exponent_dim - 1)
        ) // 2

        # Attention mask - inputs are allowed to see inputs, including future inputs
        # but not outputs; Outputs are allowed to see inputs, and previous outputs
        mask_size = 2 ** (config.log2_max_block_size + 1)
        half_mask_size = 2**config.log2_max_block_size

        # Create a mask of ones
        self.standard_attn_mask = torch.ones(mask_size, mask_size, requires_grad=False)

        # Set the lower triangular portion of the second half of the attention mask to 0
        output_mask = torch.tril(torch.ones(half_mask_size, half_mask_size))
        self.standard_attn_mask[half_mask_size:, half_mask_size:] = output_mask

        # Set the second quarter of the attention mask to 0
        self.standard_attn_mask[:half_mask_size, half_mask_size:] = 0

        # Position encoding w/ half harmonics
        # +1 for marking processed inputs
        # +1 for vacant inputs
        # +1 for marking latent (1) or vocab (0)
        # +1 for target length ratio
        # +1 for target length accuracy cost
        # +1 for shifts (for actions)
        self.reserved_inputs_dim = (config.log2_max_block_size * 4) + 6
        self.total_transformer_layers = (
            config.n_common_layers + config.n_advantage_layers
        )
        self.block_size = 2**config.log2_max_block_size
        self.look_ahead = self.block_size // 2

        transformer_dict = dict(
            vocab_embedding_1=nn.Embedding(
                config.input_vocab_size,
                4 * config.n_embd - self.reserved_inputs_dim,
            ),
            vocab_embedding_2=nn.Linear(
                4 * config.n_embd, config.n_embd - self.reserved_inputs_dim
            ),
            latent_embedding_1=nn.Linear(
                config.actor_exponent_dim**2,
                config.n_embd * config.actor_exponent_dim - self.reserved_inputs_dim,
            ),
            latent_embedding_2=nn.Linear(
                config.n_embd * config.actor_exponent_dim,
                config.n_embd - self.reserved_inputs_dim,
            ),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(
                [
                    Block(config, custom_attn_mask=self.standard_attn_mask)
                    for _ in range(self.total_transformer_layers)
                ]
            ),
            # +2 - expected reward value and variance, *2 - mean and variance, +1 - shift
            vocab_to_latent_actor=nn.Linear(
                config.n_embd, 2 + self.latent_actor_dim * 2 + 1
            ),
            # +2 - expected reward value and variance, +1 - shift
            latent_to_vocab_actor=nn.Linear(
                config.n_embd, 2 + config.input_vocab_size + 1
            ),
            # First use the vocab_embedding / latent_embedding to get to config.n_embd
            advantage_merger=nn.Linear(config.n_embd, config.n_embd),
            # Mean and variance of the advantage
            vocab_to_latent_advantage=nn.Linear(config.n_embd, 2),
            # Mean and variance of the advantage
            latent_to_vocab_advantage=nn.Linear(config.n_embd, 2),
        )

        self.transformer = nn.ModuleDict(transformer_dict)
        self.old_transformer = nn.ModuleDict(copy.deepcopy(transformer_dict))
        self.update_old_policy()
        self.frequencies_block = frequencies_block(config.log2_max_block_size)

    def create_reserved_input_dims_no_shifts(
        self,
        num_processed_items: int,
        total_items: int,
        is_latent: bool,
        target_length_ratio: float,
        target_length_accuracy_cost: float,
    ):
        processed_vector = torch.cat(
            [torch.ones(self.look_ahead), torch.zeros(self.look_ahead)]
        )
        populated_before = min(num_processed_items, self.look_ahead)
        populated_after = min(total_items - num_processed_items, self.look_ahead)
        vacant_vector = torch.cat(
            [
                torch.zeros(self.look_ahead - populated_before),
                torch.ones(populated_before + populated_after),
                torch.zeros(self.look_ahead - populated_after),
            ]
        )
        is_latent_vector = torch.zeros(self.look_ahead * 2)
        if is_latent:
            is_latent_vector = torch.ones(self.look_ahead * 2)
        target_length_ratio_vector = (
            torch.ones(self.look_ahead * 2) * target_length_ratio
        )
        target_length_accuracy_cost_vector = (
            torch.ones(self.look_ahead * 2) * target_length_accuracy_cost
        )
        shift_vector = torch.zeros(self.look_ahead * 2)
        return torch.cat(
            [
                self.frequencies_block,
                processed_vector.unsqueeze(1),
                vacant_vector.unsqueeze(1),
                is_latent_vector.unsqueeze(1),
                target_length_ratio_vector.unsqueeze(1),
                target_length_accuracy_cost_vector.unsqueeze(1),
                shift_vector.unsqueeze(1),
            ],
            dim=1,
        )

    def create_vocab_embedding(
        self,
        vocab_actions: list[LatentToVocabAction],
        num_processed_items: int,
        target_length_ratio: float,
        target_length_accuracy_cost: float,
    ):
        num_relevant_vocab_actions_before = min(num_processed_items, self.look_ahead)
        num_relevant_vocab_actions_after = min(
            len(vocab_actions) - num_processed_items, self.look_ahead
        )
        # Create reserved input dimensions
        reserved_input_dims = self.create_reserved_input_dims_no_shifts(
            num_processed_items,
            len(vocab_actions),
            False,
            target_length_ratio,
            target_length_accuracy_cost,
        )
        for i in range(
            num_processed_items - num_relevant_vocab_actions_before,
            num_processed_items + num_relevant_vocab_actions_after,
        ):
            if vocab_actions[i].shift:
                reserved_input_dims[i - num_processed_items][-1] = 1
        if len(vocab_actions) == 0:
            input_vocab_embeddings = torch.zeros(
                self.look_ahead * 2, self.config.n_embd * 4 - self.reserved_inputs_dim
            )
        else:
            input_vocab_embeddings = torch.stack(
                [
                    torch.zeros(self.config.n_embd * 4 - self.reserved_inputs_dim)
                    if action.shift
                    else self.transformer.vocab_embedding_1(
                        torch.tensor(action.vocab, dtype=torch.int64)
                    )
                    for action in vocab_actions
                ],
                dim=0,
            )
            padding_before = torch.zeros(
                self.look_ahead - num_relevant_vocab_actions_before,
                self.config.n_embd * 4 - self.reserved_inputs_dim,
            )
            padding_after = torch.zeros(
                self.look_ahead - num_relevant_vocab_actions_after,
                self.config.n_embd * 4 - self.reserved_inputs_dim,
            )
            input_vocab_embeddings = torch.cat(
                [padding_before, input_vocab_embeddings, padding_after], dim=0
            )
        input_vocab_embeddings = torch.cat(
            [input_vocab_embeddings, reserved_input_dims], dim=1
        )
        input_vocab_embeddings = torch.cat(
            [
                self.transformer.vocab_embedding_2(input_vocab_embeddings),
                reserved_input_dims,
            ],
            dim=1,
        )
        return input_vocab_embeddings

    def create_latent_embedding(
        self,
        latent_actions: list[VocabToLatentAction],
        num_processed_items: int,
        target_length_ratio: float,
        target_length_accuracy_cost: float,
    ):
        num_relevant_latent_actions_before = min(num_processed_items, self.look_ahead)
        num_relevant_latent_actions_after = min(
            len(latent_actions) - num_processed_items, self.look_ahead
        )
        # Create reserved input dimensions
        reserved_input_dims = self.create_reserved_input_dims_no_shifts(
            num_processed_items,
            len(latent_actions),
            True,
            target_length_ratio,
            target_length_accuracy_cost,
        )
        for i in range(
            num_processed_items - num_relevant_latent_actions_before,
            num_processed_items + num_relevant_latent_actions_after,
        ):
            if latent_actions[i].shift:
                reserved_input_dims[i - num_processed_items][-1] = 1
        if len(latent_actions) == 0:
            input_latent_embeddings = torch.zeros(
                self.look_ahead * 2,
                self.config.n_embd * self.config.actor_exponent_dim
                - self.reserved_inputs_dim,
            )
        else:
            input_latent_embeddings = torch.cat(
                [
                    torch.zeros(
                        self.config.n_embd * self.config.actor_exponent_dim
                        - self.reserved_inputs_dim
                    ).unsqueeze(0)
                    if action.shift
                    else self.transformer.latent_embedding_1(action.latent).unsqueeze(0)
                    for action in latent_actions
                ],
                dim=0,
            )
            padding_before = torch.zeros(
                self.look_ahead - num_relevant_latent_actions_before,
                self.config.n_embd * self.config.actor_exponent_dim
                - self.reserved_inputs_dim,
            )
            padding_after = torch.zeros(
                self.look_ahead - num_relevant_latent_actions_after,
                self.config.n_embd * self.config.actor_exponent_dim
                - self.reserved_inputs_dim,
            )
            input_latent_embeddings = torch.cat(
                [padding_before, input_latent_embeddings, padding_after], dim=0
            )
        input_latent_embeddings = torch.cat(
            [input_latent_embeddings, reserved_input_dims], dim=1
        )
        input_latent_embeddings = torch.cat(
            [
                self.transformer.latent_embedding_2(input_latent_embeddings),
                reserved_input_dims,
            ],
            dim=1,
        )
        return input_latent_embeddings

    @torch.no_grad()
    def sample_vocab_to_latent(
        self,
        vocab: list[int],
        target_length_ratio: float,
        target_length_importance: float,
        temperature: float = 1.0,
    ):
        total_items = len(vocab)
        vocab_actions = [LatentToVocabAction(vocab=i) for i in vocab]
        output_actions: list[VocabToLatentAction] = []
        processed_items = 0
        while processed_items < total_items:
            input_embeddings = self.create_vocab_embedding(
                vocab_actions,
                processed_items,
                target_length_ratio,
                target_length_importance,
            )
            output_embeddings = self.create_latent_embedding(
                output_actions,
                len(output_actions),  # All actions are "passed"
                target_length_ratio,
                target_length_importance,
            )
            x = torch.cat([input_embeddings, output_embeddings], dim=0).unsqueeze(0)
            for i in range(self.config.n_common_layers):
                x = self.transformer.h[i](x)
            actor_policy = self.transformer.vocab_to_latent_actor(x)[0][
                self.look_ahead * 3
            ]
            sampler = torch.normal(0.0, 1.0, size=(self.latent_actor_dim,))
            shift_sampler = torch.rand(1)[0]
            latent_so_means = actor_policy[2 : 2 + self.latent_actor_dim]
            latent_so_vars = (
                torch.exp(
                    actor_policy[
                        2 + self.latent_actor_dim : 2 + 2 * self.latent_actor_dim
                    ]
                )
                * temperature
            )
            shift = torch.sigmoid(actor_policy[-1] / temperature)
            if shift_sampler < shift:
                new_action = VocabToLatentAction(shift=True)
                processed_items += 1
            else:
                latent_exponentiated = so_to_SO(
                    self.config.actor_exponent_dim,
                    latent_so_means + sampler * latent_so_vars,
                )
                new_action = VocabToLatentAction(latent=latent_exponentiated)
            output_actions.append(new_action)
        return output_actions

    @torch.no_grad()
    def sample_latent_to_vocab(
        self,
        latents: list[torch.Tensor],
        target_length_ratio: float,
        target_length_importance: float,
        temperature: float = 1.0,
    ):
        total_items = len(latents)
        latent_actions = [VocabToLatentAction(latent=i) for i in latents]
        output_actions: list[LatentToVocabAction] = []
        processed_items = 0
        while processed_items < total_items:
            input_embeddings = self.create_latent_embedding(
                latent_actions,
                processed_items,
                target_length_ratio,
                target_length_importance,
            )
            output_embeddings = self.create_vocab_embedding(
                output_actions,
                len(output_actions),  # All actions are "passed"
                target_length_ratio,
                target_length_importance,
            )
            x = torch.cat([input_embeddings, output_embeddings], dim=0).unsqueeze(0)
            for i in range(self.config.n_common_layers):
                x = self.transformer.h[i](x)
            actor_policy = self.transformer.latent_to_vocab_actor(x)[0][
                self.look_ahead * 3
            ]
            shift_sampler = torch.rand(1)[0]
            vocab_pre_softmax = (
                actor_policy[2 : 2 + self.config.input_vocab_size] / temperature
            )
            shift = torch.sigmoid(actor_policy[-1] / temperature)
            if shift_sampler < shift:
                new_action = LatentToVocabAction(shift=True)
                processed_items += 1
            else:
                softmax_vocab = torch.distributions.categorical.Categorical(
                    torch.softmax(vocab_pre_softmax, dim=0)
                )
                sampled = softmax_vocab.sample()
                new_action = LatentToVocabAction(vocab=sampled)
            output_actions.append(new_action)
        return output_actions

    def update_old_policy(self):
        self.old_transformer.load_state_dict(
            copy.deepcopy(self.transformer.state_dict())
        )

    def latent_kl_divergence(self, old_policy_tensor, new_policy_tensor):
        old_mean = old_policy_tensor[:, 2 : 2 + self.latent_actor_dim]
        old_var = torch.exp(
            old_policy_tensor[
                :, 2 + self.latent_actor_dim : 2 + 2 * self.latent_actor_dim
            ]
        )
        old_shift = torch.sigmoid(old_policy_tensor[:, -1])
        new_mean = new_policy_tensor[:, 2 : 2 + self.latent_actor_dim]
        new_var = torch.exp(
            new_policy_tensor[
                :, 2 + self.latent_actor_dim : 2 + 2 * self.latent_actor_dim
            ]
        )
        new_shift = torch.sigmoid(new_policy_tensor[:, -1])
        continuous_kl = 0.5 * (
            torch.log(new_var.norm() / old_var.norm())
            # Ignore dimension constant
            + ((new_mean - old_mean) / new_var).dot(new_mean - old_mean)
            + (old_var / new_var).sum(dim=0)
        )
        shift_kl = old_shift * torch.log(old_shift / new_shift)
        kl = shift_kl + (1 - old_shift) * (
            continuous_kl + torch.log((1 - old_shift) / (1 - new_shift))
        )
        return kl

    def vocab_kl_divergence(self, old_policy_tensor, new_policy_tensor):
        old_shift = torch.sigmoid(old_policy_tensor[:, -1])
        new_shift = torch.sigmoid(new_policy_tensor[:, -1])
        shift_kl = old_shift * torch.log(old_shift / new_shift)
        softmax_old = torch.softmax(
            old_policy_tensor[:, 2 : 2 + self.config.input_vocab_size], dim=1
        )
        softmax_new = torch.softmax(
            new_policy_tensor[:, 2 : 2 + self.config.input_vocab_size], dim=1
        )
        softmax_kl = torch.sum(softmax_old * torch.log(softmax_old / softmax_new))
        return shift_kl + (1 - old_shift) * (
            softmax_kl + torch.log((1 - old_shift) / (1 - new_shift))
        )

    def ppo_train(
        self,
        recorded_playthroughs: list[GPTAutoTokenizerRecorderSession],
        beta: float,
        target_kl_divergence: float,
        num_dataset_iterations: int,
        max_grad_steps: int,
    ) -> float:
        """
        Return value: current kl divergence from previous policy
        """
        random.shuffle(recorded_playthroughs)
        split_idx = int(len(recorded_playthroughs) * self.config.val_split_percentile)
        val_playthroughs = recorded_playthroughs[:split_idx]
        train_playthroughs = recorded_playthroughs[split_idx:]
        total_rewards_over_length = [
            rp.total_reward_over_length for rp in recorded_playthroughs
        ]
        total_rewards_over_length_with_indices = enumerate(total_rewards_over_length)
        # Reinforce good existing behavior
        total_rewards_over_length_with_start_indices = dict(
            sorted(
                total_rewards_over_length_with_indices,
                key=lambda x: x[1],
            )
        )
        total_rewards_over_length_with_end_indices = dict(
            sorted(
                total_rewards_over_length_with_indices[::-1],
                key=lambda x: x[1],
            )
        )
        total_reward_std = np.std(total_rewards_over_length)
        self.update_old_policy()
        optimizer = torch.optim.Adam(
            self.transformer.parameters(), lr=self.config.a2c_learning_rate
        )
        kl_divergence = 0
        while (
            kl_divergence / target_kl_divergence
            < self.config.early_stop_kl_divergence_target_factor
        ):
            pass
        return 0.1
