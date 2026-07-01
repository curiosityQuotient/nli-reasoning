"""GRPO training configuration."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.models.config import (
    BETA,
    EPSILON,
    LEARNING_RATE,
    MAX_STEPS,
    NUM_GENERATIONS,
    NUM_ITERATIONS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    B1,
    B2,
)


@dataclass
class GRPOConfig:
    """Configuration for Group Relative Policy Optimization.
    
    Attributes:
        num_generations: Number of responses per prompt (G in GRPO)
        num_iterations: Iterations per batch
        beta: KL divergence penalty coefficient
        epsilon: PPO clipping parameter
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping threshold
        warmup_steps: Warmup steps for learning rate schedule
        max_steps: Total training steps
        b1: Adam beta1
        b2: Adam beta2
    """
    num_generations: int = NUM_GENERATIONS
    num_iterations: int = NUM_ITERATIONS
    beta: float = BETA
    epsilon: float = EPSILON
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_grad_norm: float = MAX_GRAD_NORM
    warmup_steps: int = WARMUP_STEPS
    max_steps: int = MAX_STEPS
    b1: float = B1
    b2: float = B2
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "num_generations": self.num_generations,
            "num_iterations": self.num_iterations,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "b1": self.b1,
            "b2": self.b2,
        }


def create_grpo_config_stage1() -> GRPOConfig:
    """Create GRPO config for Stage 1 (GSM8K training).
    
    Returns:
        GRPOConfig for stage 1
    """
    return GRPOConfig(
        num_generations=4,
        num_iterations=1,
        beta=0.08,
        epsilon=0.2,
    )


def create_grpo_config_stage2() -> GRPOConfig:
    """Create GRPO config for Stage 2 (NLI training with uncertainty).
    
    Returns:
        GRPOConfig for stage 2
    """
    return GRPOConfig(
        num_generations=4,
        num_iterations=2,
        beta=0.08,
        epsilon=0.2,
    )
