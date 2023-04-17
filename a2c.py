import math
import random
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
    # Biases
    bias: bool = True


@dataclass
class A2CGPTEncoderAction:
    shift: bool
    sampled_action: Optional[torch.Tensor]


@dataclass
class ExpectedMeanVar:
    mean: torch.Tensor
    var: torch.Tensor


@dataclass
class A2CGPTEncoderLossProjection:
    actor_entropy_loss: torch.Tensor
    critic_value_pre_actor: ExpectedMeanVar
    critic_value_post_actor: ExpectedMeanVar
    advantage_value: ExpectedMeanVar


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


class A2CGPTEncoderModel(nn.Module):
    def __init__(self, config: A2CGPTEncoderConfig):
        super().__init__()
        self.config = config

        assert (
            self.config.actor_exponent_dim * (self.config.actor_exponent_dim - 1) // 2
            == self.config.actor_latent_dim
        )

        # Attention mask - inputs are allowed to see inputs, including future inputs
        # but not outputs; Outputs are allowed to see inputs, and previous outputs
        # Efficient mask:
        mask_size = 2 ** (config.log2_max_block_size + 1)
        half_mask_size = 2**config.log2_max_block_size

        # Create a mask of ones
        self.standard_attn_mask = torch.ones(mask_size, mask_size, requires_grad=False)

        # Set the lower triangular portion of the second half of the attention mask to 0
        output_mask = torch.tril(torch.ones(half_mask_size, half_mask_size))
        self.standard_attn_mask[half_mask_size:, half_mask_size:] = output_mask

        # Set the second quarter of the attention mask to 0
        self.standard_attn_mask[:half_mask_size, half_mask_size:] = 0

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
                output_translation=nn.Linear(
                    # Source action
                    config.actor_latent_dim
                    # Exponented action in SO(5)
                    + config.actor_exponent_dim * config.actor_exponent_dim
                    # + position (8 * log2_size)
                    + config.log2_max_block_size * 4
                    # + already_encoded, vacancy, shift, output indicator
                    + 4,
                    config.n_embd - (config.log2_max_block_size * 4 + 4),
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(config, custom_attn_mask=self.standard_attn_mask)
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
                    + config.actor_exponent_dim * config.actor_exponent_dim
                    + 1,  # +1 for shift operation
                    config.n_embd,
                ),
                advantage_head=nn.Linear(config.n_embd, 2),  # mean and logvar
            )
        )
        self.apply(self._init_weights)
        self.frequencies_block = frequencies_block(config.log2_max_block_size)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if hasattr(module, "weight"):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
                / math.sqrt(
                    self.config.n_common_layers
                    + self.config.n_critic_layers
                    + self.config.n_advantage_layers
                ),
            )
        if hasattr(module, "bias"):
            torch.nn.init.normal_(
                module.bias,
                mean=0.0,
                std=1e-3
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
        temperature: float,
        max_actions: Optional[float] = math.inf,
        sample_losses: bool = False,
    ) -> tuple[list[A2CGPTEncoderAction], list[A2CGPTEncoderLossProjection]]:
        input_bytes = input.encode("utf-8")
        shifts_to_end = len(input_bytes)
        shifted_bytes = 0
        input_embeddings = self.transformer.wte(
            torch.tensor(np.frombuffer(input_bytes, dtype=np.uint8), dtype=torch.int64)
        )
        output_actions: list[A2CGPTEncoderAction] = []
        outpus_losses: list[A2CGPTEncoderLossProjection] = []
        num_items_ahead = 2 ** (self.config.log2_max_block_size - 1)
        while shifted_bytes < shifts_to_end and len(output_actions) < max_actions:
            (
                full_input_context,
                processed_vector,
                vacancy_vector,
            ) = self.create_input_embeddings(
                input_embeddings,
                shifted_bytes,
                num_items_ahead,
                target_len_ratio,
                target_len_importance,
            )
            output_embeddings = self.create_output_embeddings(
                output_actions, num_items_ahead
            )
            network_input = torch.unsqueeze(
                torch.cat([output_embeddings, full_input_context], dim=0), dim=0
            )
            x = network_input
            for i in range(self.config.n_common_layers):
                x = self.transformer.h[i](x)
            actor_policy = self.transformer.actor_head(x)[0][num_items_ahead * 3]
            pre_actor_value = actor_policy[0]
            pre_actor_value_var = torch.exp(actor_policy[1])
            policy_token_mean = actor_policy[2 : 2 + self.config.actor_latent_dim]
            policy_logvar = actor_policy[
                2 + self.config.actor_latent_dim : 2 + 2 * self.config.actor_latent_dim
            ]
            policy_token_var = torch.exp(policy_logvar * temperature)
            policy_shift = torch.sigmoid(actor_policy[-1] * temperature)
            sampled_random = random.random()
            action = self.sample_action(
                policy_shift,
                policy_token_mean,
                policy_token_var,
                sampled_random,
            )
            if action.shift:
                shifted_bytes = shifted_bytes + 1

            output_actions.append(action)

            if sample_losses:
                critic_merger_input = self.create_critic_merger_input(
                    x,
                    num_items_ahead,
                    policy_token_mean,
                    policy_token_var,
                    policy_shift,
                )
                x = x + self.transformer.critic_merger(critic_merger_input)
                for i in range(self.config.n_critic_layers):
                    x = self.transformer.h[self.config.n_common_layers + i](x)
                critic_result = self.transformer.critic_head(x)[0][num_items_ahead * 3]
                critic_mean = critic_result[0]
                critic_var = torch.exp(critic_result[1])
                advantage_merger_input = self.create_advantage_merger_input(
                    x,
                    num_items_ahead,
                    policy_shift,
                    policy_token_mean,
                    policy_token_var,
                    action,
                )
                x = x + self.transformer.advantage_merger(advantage_merger_input)
                for i in range(self.config.n_advantage_layers):
                    x = self.transformer.h[
                        self.config.n_common_layers + self.config.n_critic_layers + i
                    ](x)
                advantage_result = self.transformer.advantage_head(x)[0][
                    num_items_ahead * 3
                ]
                advantage_mean = advantage_result[0]
                advantage_var = torch.exp(advantage_result[1])

            outpus_losses.append(
                A2CGPTEncoderLossProjection(
                    actor_entropy_loss=torch.sum(policy_logvar**2),
                    critic_value_pre_actor=ExpectedMeanVar(
                        mean=pre_actor_value, var=pre_actor_value_var
                    ),
                    critic_value_post_actor=ExpectedMeanVar(
                        mean=0 if not sample_losses else critic_mean,
                        var=0 if not sample_losses else critic_var,
                    ),
                    advantage_value=ExpectedMeanVar(
                        mean=0 if not sample_losses else advantage_mean,
                        var=0 if not sample_losses else advantage_var,
                    ),
                )
            )
        return output_actions, outpus_losses

    def create_input_embeddings(
        self,
        input_embeddings,
        shifted_bytes,
        num_items_ahead,
        target_len_ratio,
        target_len_importance,
    ):
        processed_visible_bytes = min(shifted_bytes, num_items_ahead)
        input_processed = input_embeddings[
            shifted_bytes - processed_visible_bytes : shifted_bytes
        ]
        input_context = input_embeddings[
            shifted_bytes : shifted_bytes + num_items_ahead
        ]
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
        processed_vector = torch.cat(
            [torch.ones(num_items_ahead), torch.zeros(num_items_ahead)]
        )
        vacancy_vector = torch.cat(
            [
                torch.zeros(num_items_ahead - input_processed.shape[0]),
                torch.ones(input_processed.shape[0]),
                torch.ones(input_context.shape[0]),
                torch.zeros(num_items_ahead - input_context.shape[0]),
            ]
        )
        full_input_context = torch.cat(
            [
                full_input_context,
                self.frequencies_block,
                torch.unsqueeze(processed_vector, dim=1),
                torch.unsqueeze(vacancy_vector, dim=1),
                torch.unsqueeze(
                    torch.ones(num_items_ahead * 2) * target_len_ratio, dim=1
                ),
                torch.unsqueeze(
                    torch.ones(num_items_ahead * 2) * target_len_importance, dim=1
                ),
                torch.unsqueeze(torch.zeros(num_items_ahead * 2), dim=1),
            ],
            dim=1,
        )
        return full_input_context, processed_vector, vacancy_vector

    def create_output_embeddings(self, output_actions, num_items_ahead):
        relevant_actions = output_actions[-num_items_ahead:]
        output = torch.zeros(
            num_items_ahead * 2,
            self.config.actor_latent_dim
            + self.config.actor_exponent_dim * self.config.actor_exponent_dim,
        )
        processed_vector = torch.cat(
            [torch.ones(num_items_ahead), torch.zeros(num_items_ahead)]
        )
        vacancy_vector = torch.cat(
            [
                torch.zeros(num_items_ahead - len(relevant_actions)),
                torch.ones(len(relevant_actions)),
                torch.zeros(num_items_ahead),
            ]
        )
        output = torch.cat(
            [
                output,
                self.frequencies_block,
                torch.unsqueeze(processed_vector, dim=1),
                torch.unsqueeze(vacancy_vector, dim=1),
                torch.unsqueeze(torch.zeros(num_items_ahead * 2), dim=1),
                torch.unsqueeze(torch.ones(num_items_ahead * 2), dim=1),
            ],
            dim=1,
        )
        ptr = max(len(relevant_actions) - num_items_ahead, 0)
        while ptr < len(relevant_actions):
            action = relevant_actions[ptr]
            if action.shift:
                output[len(relevant_actions) - ptr - 1, -2] = 1
            else:
                exponentiated_action = so_to_SO(
                    self.config.actor_exponent_dim, action.sampled_action
                )
                output[
                    len(relevant_actions) - ptr, : self.config.actor_latent_dim
                ] = action.sampled_action
                output[
                    len(relevant_actions) - ptr - 1,
                    self.config.actor_latent_dim : self.config.actor_latent_dim
                    + self.config.actor_exponent_dim * self.config.actor_exponent_dim,
                ] = exponentiated_action
            ptr += 1
        output_embeddings = self.transformer.output_translation(output)
        output_embeddings = torch.cat(
            [
                output_embeddings,
                self.frequencies_block,
                torch.unsqueeze(
                    torch.cat(
                        [torch.ones(num_items_ahead), torch.zeros(num_items_ahead)]
                    ),
                    dim=1,
                ),
                torch.unsqueeze(vacancy_vector, dim=1),
                torch.unsqueeze(torch.zeros(num_items_ahead * 2), dim=1),
                torch.unsqueeze(torch.ones(num_items_ahead * 2), dim=1),
            ],
            dim=1,
        )
        return output_embeddings

    def create_critic_merger_input(
        self, x, num_items_ahead, policy_token_mean, policy_token_var, policy_shift
    ):
        critic_merger_input = torch.unsqueeze(
            torch.cat(
                [
                    x[0],
                    torch.zeros(
                        num_items_ahead * 4,
                        self.config.actor_latent_dim * 2 + 1,
                    ),
                ],
                dim=1,
            ),
            dim=0,
        )
        critic_merger_input[0][num_items_ahead * 3][self.config.n_embd :] = torch.cat(
            [
                policy_token_mean,
                policy_token_var,
                torch.unsqueeze(policy_shift, dim=0),
            ],
            dim=0,
        )
        return critic_merger_input

    # def create_advantage_merger_input_and_action(
    #     self,
    #     x,
    #     num_items_ahead,
    #     policy_shift,
    #     policy_token_mean,
    #     policy_token_var,
    #     sampled_random,
    # ):
    #     advantage_merger_input = torch.unsqueeze(
    #         torch.cat(
    #             [
    #                 x[0],
    #                 torch.zeros(
    #                     num_items_ahead * 4,
    #                     self.config.actor_latent_dim
    #                     + self.config.actor_exponent_dim
    #                     * self.config.actor_exponent_dim
    #                     + 1,
    #                 ),
    #             ],
    #             dim=1,
    #         ),
    #         dim=0,
    #     )
    #     if sampled_random < policy_shift:
    #         action = A2CGPTEncoderAction(shift=True, sampled_action=None)
    #         advantage_merger_input[0][num_items_ahead * 3][-1] = 1
    #     else:
    #         sampler = torch.normal(0.0, 1.0, size=(self.config.actor_latent_dim,))
    #         action = A2CGPTEncoderAction(
    #             shift=False,
    #             sampled_action=policy_token_mean + sampler * policy_token_var,
    #         )
    #         advantage_merger_input[0][num_items_ahead * 3][
    #             -self.config.actor_latent_dim
    #             - self.config.actor_exponent_dim * self.config.actor_exponent_dim
    #             - 1 : -self.config.actor_exponent_dim * self.config.actor_exponent_dim
    #             - 1
    #         ] = action.sampled_action
    #         advantage_merger_input[0][num_items_ahead * 3][
    #             -self.config.actor_exponent_dim * self.config.actor_exponent_dim
    #             - 1 : -1
    #         ] = so_to_SO(self.config.actor_exponent_dim, action.sampled_action)
    #     return advantage_merger_input, action

    def create_advantage_merger_input(
        self,
        x,
        num_items_ahead,
        policy_shift,
        policy_token_mean,
        policy_token_var,
        action,
    ):
        advantage_merger_input = torch.unsqueeze(
            torch.cat(
                [
                    x[0],
                    torch.zeros(
                        num_items_ahead * 4,
                        self.config.actor_latent_dim
                        + self.config.actor_exponent_dim
                        * self.config.actor_exponent_dim
                        + 1,
                    ),
                ],
                dim=1,
            ),
            dim=0,
        )
        if action.shift:
            advantage_merger_input[0][num_items_ahead * 3][-1] = 1
        else:
            advantage_merger_input[0][num_items_ahead * 3][
                -self.config.actor_latent_dim
                - self.config.actor_exponent_dim * self.config.actor_exponent_dim
                - 1 : -self.config.actor_exponent_dim * self.config.actor_exponent_dim
                - 1
            ] = action.sampled_action
            advantage_merger_input[0][num_items_ahead * 3][
                -self.config.actor_exponent_dim * self.config.actor_exponent_dim
                - 1 : -1
            ] = so_to_SO(self.config.actor_exponent_dim, action.sampled_action)
        return advantage_merger_input

    def sample_action(
        self,
        policy_shift,
        policy_token_mean,
        policy_token_var,
        sampled_random,
    ):
        if sampled_random < policy_shift:
            action = A2CGPTEncoderAction(shift=True, sampled_action=None)
        else:
            sampler = torch.normal(0.0, 1.0, size=(self.config.actor_latent_dim,))
            action = A2CGPTEncoderAction(
                shift=False,
                sampled_action=policy_token_mean + sampler * policy_token_var,
            )
        return action
