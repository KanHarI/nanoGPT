from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

import torch

from a2c.text_loss import similar


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
    learning_rate: float = 1e-4
    # Auxilary loss - reinforce patterns that resulted in good total reward,
    # even if missed by critic. Important early on, when critic is not
    # accurate.
    learning_rate_warmup_steps: int = 500
    stability_relative_weight_to_reward_std: float = 0.1
    # Decay rate of stability reward between PPO iterations
    stability_decay_rate: float = 0.85
    # Reinforce the top half of the reward distribution, with bigger rewards
    # by percentiles (linearly from the min_percentile_stability_reward to 1)
    min_percentile_stability_reward: float = 0.5
    # Factor by which the KL divergence must be less than the target, otherwise
    # stop PPO training early
    early_stop_kl_divergence_target_factor: float = 2.0
    # Update the current approximated training KL after ~20 batches
    training_kl_moving_average_iterations_constant: float = 0.95
    # Discount factor for the reward
    discount_factor: float = 0.9
    # Validation split percentile
    val_split_percentile: float = 0.2
    # Allowed critic overfit
    # If validation set loss (for either value or advantage networks) is more than
    # the best observed validation loss multiplied by this factor, then stop PPO
    # training early
    critic_overfit_allowed: float = 1.1
    # Iterations on validation set to determine if overfitting and early stop
    # or it KL divergence is getting too high
    val_num_iterations: int = 100
    # Maximum number of gradient steps
    max_grad_steps: int = 5000
    # Number of iterations to run PPO before running validation
    validate_every: int = 100


class PPOStopingReason(Enum):
    KL_TOO_LARGE = 1
    CRITIC_OVERFIT = 2
    OUT_OF_DATASET_ITERATIONS = 3
    MAX_GRAD_STEPS = 4


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
    # L_v
    input_vocab: torch.Tensor
    # (L_v + L_l)
    encoder_extra_reward: torch.Tensor
    # (L_v + L_l) [booleans]
    encoder_shifts: torch.Tensor
    # L_l * (E * (E - 1) / 2)
    latent_representation: torch.Tensor
    # L_l * (E ** 2)
    exponent_representation: torch.Tensor
    # (L_l + L_v2)
    decoder_reward: torch.Tensor
    # (L_l + L_v2) [booleans]
    decoder_shifts: torch.Tensor
    # (1)
    total_reward: float
    total_reward_over_length: float
