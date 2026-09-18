"""GRPO trainer wrapper for two-stage learning.

Wraps the Tunix 0.1.7 RL stack (``RLCluster`` / ``GRPOLearner``) behind a
small interface for the two-stage GSM8K -> NLI pipeline.
"""

import gc
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh

from src.models.config import (
    B1,
    B2,
    CHECKPOINT_MAX_TO_KEEP,
    EVAL_EVERY_N_STEPS,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_PROMPT_LENGTH,
    MAX_STEPS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    TOTAL_GENERATION_STEPS,
    TRAIN_MICRO_BATCH_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from src.training.grpo_config import GRPOConfig


def create_optimizer(
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    max_grad_norm: float = MAX_GRAD_NORM,
    warmup_steps: int = WARMUP_STEPS,
    max_steps: int = MAX_STEPS,
    b1: float = B1,
    b2: float = B2,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with warmup cosine decay schedule.

    Args:
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps
        max_steps: Total number of training steps
        b1: Adam beta1
        b2: Adam beta2

    Returns:
        Optimizer transformation
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max_steps,
        end_value=0.0,
    )

    optimizer = optax.adamw(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        weight_decay=weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=max_grad_norm),
        optimizer,
    )

    return optimizer


class GRPOTrainer:
    """Wrapper for Group Relative Policy Optimization training.

    Builds a Tunix ``RLCluster`` (actor = policy model, reference =
    frozen base model, vanilla rollout engine) and a ``GRPOLearner`` for
    the given reward functions.
    """

    def __init__(
        self,
        model: nnx.Module,
        ref_model: nnx.Module,
        optimizer: optax.GradientTransformation,
        grpo_config: GRPOConfig,
        reward_fns: List[Callable],
        tokenizer: Any,
        mesh: Mesh,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        eval_every_n_steps: int = EVAL_EVERY_N_STEPS,
        use_wandb: bool = False,
    ):
        """Initialize GRPO trainer.

        Args:
            model: Policy model with LoRA applied
            ref_model: Reference model (frozen)
            optimizer: Optimizer transformation
            grpo_config: GRPO configuration
            reward_fns: List of reward functions
            tokenizer: Gemma tokenizer
            mesh: JAX mesh for sharding
            checkpoint_dir: Directory for Tunix-managed checkpoints
            eval_every_n_steps: Steps between evaluations
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.grpo_config = grpo_config
        self.reward_fns = reward_fns
        self.tokenizer = tokenizer
        self.mesh = mesh
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.eval_every_n_steps = eval_every_n_steps
        self.use_wandb = use_wandb

    def _build_cluster_config(self, total_steps: int) -> Any:
        """Build the Tunix cluster configuration for one training stage.

        Args:
            total_steps: Total training steps for this stage

        Returns:
            Configured ``ClusterConfig``
        """
        from tunix.rl import rl_cluster as rl_cluster_lib
        from tunix.rl.rollout import base_rollout

        checkpoint_root = (
            str(self.checkpoint_dir) if self.checkpoint_dir is not None else None
        )
        checkpointing_options = (
            ocp.CheckpointManagerOptions(
                save_interval_steps=total_steps,
                max_to_keep=CHECKPOINT_MAX_TO_KEEP,
            )
            if checkpoint_root is not None
            else None
        )

        return rl_cluster_lib.ClusterConfig(
            role_to_mesh={
                rl_cluster_lib.Role.ACTOR: self.mesh,
                rl_cluster_lib.Role.REFERENCE: self.mesh,
                rl_cluster_lib.Role.ROLLOUT: self.mesh,
            },
            rollout_engine="vanilla",
            offload_to_cpu=False,
            training_config=rl_cluster_lib.RLTrainingConfig(
                actor_optimizer=self.optimizer,
                eval_every_n_steps=self.eval_every_n_steps,
                max_steps=total_steps,
                mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
                train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
                checkpoint_root_directory=checkpoint_root,
                checkpointing_options=checkpointing_options,
            ),
            rollout_config=base_rollout.RolloutConfig(
                max_tokens_to_generate=TOTAL_GENERATION_STEPS,
                max_prompt_length=MAX_PROMPT_LENGTH,
                kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            ),
        )

    def train(
        self,
        dataset: Any,
        total_steps: Optional[int] = None,
        wandb_run_name: Optional[str] = None,
    ) -> None:
        """Execute GRPO training.

        The policy model is updated in place by Tunix; checkpoints (if
        configured) are written under ``checkpoint_dir``.

        Args:
            dataset: Training dataset of prompt dicts
            total_steps: Override total training steps
            wandb_run_name: Name for wandb run
        """
        from tunix.rl.grpo.grpo_learner import GRPOConfig as TunixGRPOConfig
        from tunix.rl.grpo.grpo_learner import GRPOLearner

        steps = total_steps or self.grpo_config.max_steps
        cluster_config = self._build_cluster_config(steps)

        rl_cluster = self._rl_cluster_lib.RLCluster(
            actor=self.model,
            reference=self.ref_model,
            tokenizer=self.tokenizer,
            cluster_config=cluster_config,
        )

        grpo_config = TunixGRPOConfig(
            num_generations=self.grpo_config.num_generations,
            num_iterations=self.grpo_config.num_iterations,
            beta=self.grpo_config.beta,
            epsilon=self.grpo_config.epsilon,
        )

        grpo_learner = GRPOLearner(
            rl_cluster=rl_cluster,
            algo_config=grpo_config,
            reward_fns=self.reward_fns,
        )

        if self.use_wandb:
            import wandb

            if wandb.run is None:
                wandb.init(
                    name=wandb_run_name,
                    config=self.grpo_config.to_dict(),
                )

        with self.mesh:
            grpo_learner.train(dataset)

        if self.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.finish()

        gc.collect()


def train_two_stage(
    model: nnx.Module,
    ref_model: nnx.Module,
    tokenizer: Any,
    mesh: Mesh,
    train_gsm8k_dataset: Any,
    train_nli_dataset: Any,
    grpo_config_stage1: Optional[GRPOConfig] = None,
    grpo_config_stage2: Optional[GRPOConfig] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    use_wandb: bool = False,
) -> nnx.Module:
    """Execute two-stage training pipeline.

    Stage 1: Train on GSM8K math reasoning.
    Stage 2: Continue training the same policy on NLI with uncertainty
    awareness, using a fresh optimizer.

    Each stage checkpoints into its own subdirectory, so stage 2 never
    overwrites stage 1.

    Args:
        model: Policy model
        ref_model: Reference model
        tokenizer: Gemma tokenizer
        mesh: JAX mesh for sharding
        train_gsm8k_dataset: GSM8K training dataset
        train_nli_dataset: NLI training dataset
        grpo_config_stage1: GRPO config for stage 1
        grpo_config_stage2: GRPO config for stage 2
        checkpoint_dir: Directory for checkpoints
        use_wandb: Whether to use wandb logging

    Returns:
        The trained policy model (updated in place)
    """
    from src.rewards import (
        check_answer,
        check_answer_nli,
        match_format_approximately,
        match_format_exactly,
    )

    if grpo_config_stage1 is None:
        from src.training.grpo_config import create_grpo_config_stage1

        grpo_config_stage1 = create_grpo_config_stage1()

    if grpo_config_stage2 is None:
        from src.training.grpo_config import create_grpo_config_stage2

        grpo_config_stage2 = create_grpo_config_stage2()

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None

    optimizer_stage1 = create_optimizer(
        learning_rate=grpo_config_stage1.learning_rate,
        weight_decay=grpo_config_stage1.weight_decay,
        max_grad_norm=grpo_config_stage1.max_grad_norm,
        warmup_steps=grpo_config_stage1.warmup_steps,
        max_steps=grpo_config_stage1.max_steps,
    )

    trainer_stage1 = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer_stage1,
        grpo_config=grpo_config_stage1,
        reward_fns=[match_format_exactly, match_format_approximately, check_answer],
        tokenizer=tokenizer,
        mesh=mesh,
        checkpoint_dir=checkpoint_root / "stage1" if checkpoint_root else None,
        use_wandb=use_wandb,
    )

    print("Starting Stage 1: GSM8K Training")
    trainer_stage1.train(
        train_gsm8k_dataset,
        wandb_run_name="stage1_gsm8k",
    )

    gc.collect()

    optimizer_stage2 = create_optimizer(
        learning_rate=grpo_config_stage2.learning_rate,
        weight_decay=grpo_config_stage2.weight_decay,
        max_grad_norm=grpo_config_stage2.max_grad_norm,
        warmup_steps=grpo_config_stage2.warmup_steps,
        max_steps=grpo_config_stage2.max_steps,
    )

    trainer_stage2 = GRPOTrainer(
        model=trainer_stage1.model,
        ref_model=ref_model,
        optimizer=optimizer_stage2,
        grpo_config=grpo_config_stage2,
        reward_fns=[check_answer_nli],
        tokenizer=tokenizer,
        mesh=mesh,
        checkpoint_dir=checkpoint_root / "stage2" if checkpoint_root else None,
        use_wandb=use_wandb,
    )

    print("Starting Stage 2: NLI Training with Uncertainty Awareness")
    trainer_stage2.train(
        train_nli_dataset,
        wandb_run_name="stage2_nli",
    )

    return trainer_stage2.model
