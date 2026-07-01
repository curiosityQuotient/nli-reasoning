"""GRPO trainer wrapper for two-stage learning."""

import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import optax
from flax import nnx
from jax.sharding import Mesh

from src.models.config import (
    B1,
    B2,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_STEPS,
    WEIGHT_DECAY,
    WARMUP_STEPS,
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
        weight_decay: Weight decay coefficient
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
    
    This class wraps the tunix GRPO implementation to provide
    a clean interface for two-stage learning.
    """
    
    def __init__(
        self,
        model: nnx.Module,
        ref_model: nnx.Module,
        optimizer: optax.GradientTransformation,
        grpo_config: GRPOConfig,
        reward_fns: List[Callable],
        mesh: Optional[Mesh] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_wandb: bool = False,
    ):
        """Initialize GRPO trainer.
        
        Args:
            model: Policy model with LoRA applied
            ref_model: Reference model (frozen)
            optimizer: Optimizer transformation
            grpo_config: GRPO configuration
            reward_fns: List of reward functions
            mesh: JAX mesh for sharding
            checkpoint_dir: Directory for saving checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        try:
            from tunix.rl.grpo import grpo_learner
            from tunix.rl import rl_cluster_lib
        except ImportError:
            raise ImportError(
                "tunix package not installed. "
                "Install with: pip install google-tunix[prod]"
            )
        
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.grpo_config = grpo_config
        self.reward_fns = reward_fns
        self.mesh = mesh or self._create_default_mesh()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_wandb = use_wandb
        
        self._rl_cluster = None
        self._grpo_learner = None
    
    def _create_default_mesh(self) -> Mesh:
        """Create default mesh from available devices."""
        devices = jax.local_devices()
        return Mesh(
            jax.numpy.array(devices).reshape(1, len(devices)),
            ("fsdp", "tp")
        )
    
    def _setup_rl_cluster(
        self,
        total_steps: int,
        rollout_engine: str = "vanilla"
    ) -> Any:
        """Set up RL cluster for training.
        
        Args:
            total_steps: Total training steps
            rollout_engine: Type of rollout engine
            
        Returns:
            Configured RL cluster
        """
        from tunix.rl import rl_cluster_lib
        
        cluster_config = rl_cluster_lib.ClusterConfig(
            total_steps=total_steps,
            rollout_engine=rollout_engine,
        )
        
        rl_cluster = rl_cluster_lib.RlCluster(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            cluster_config=cluster_config,
            mesh=self.mesh,
        )
        
        return rl_cluster
    
    def train(
        self,
        dataset: Any,
        total_steps: Optional[int] = None,
        wandb_run_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """Execute GRPO training.
        
        Args:
            dataset: Training dataset
            total_steps: Override total training steps
            wandb_run_name: Name for wandb run
            
        Returns:
            Dictionary of final metrics
        """
        from tunix.rl.grpo import grpo_learner
        
        steps = total_steps or self.grpo_config.max_steps
        
        self._rl_cluster = self._setup_rl_cluster(steps)
        
        grpo_config = grpo_learner.GRPOConfig(
            num_generations=self.grpo_config.num_generations,
            num_iterations=self.grpo_config.num_iterations,
            beta=self.grpo_config.beta,
            epsilon=self.grpo_config.epsilon,
        )
        
        self._grpo_learner = grpo_learner.GRPOLearner(
            rl_cluster=self._rl_cluster,
            reward_fns=self.reward_fns,
            grpo_config=grpo_config,
        )
        
        if self.use_wandb and wandb_run_name:
            import wandb
            wandb.init(
                name=wandb_run_name,
                config=self.grpo_config.to_dict(),
            )
        
        metrics = self._grpo_learner.train(dataset)
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        gc.collect()
        
        return metrics
    
    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoint.
        
        Args:
            step: Training step number
        """
        if self.checkpoint_dir is None:
            return
            
        import orbax.checkpoint as ocp
        
        checkpoint_path = self.checkpoint_dir / f"step_{step}"
        checkpointer = ocp.StandardCheckpointer()
        
        state = nnx.state(self.model)
        checkpointer.save(checkpoint_path / "model_params", state)
        checkpointer.wait_until_finished()
    
    def load_checkpoint(self, step: int) -> None:
        """Load model checkpoint.
        
        Args:
            step: Training step number to load
        """
        if self.checkpoint_dir is None:
            return
            
        import orbax.checkpoint as ocp
        
        checkpoint_path = self.checkpoint_dir / f"step_{step}"
        checkpointer = ocp.StandardCheckpointer()
        
        state = checkpointer.restore(
            checkpoint_path / "model_params",
            args=ocp.args.StandardRestore(restore_type=nnx.State)
        )
        
        graph_def, _ = nnx.split(self.model)
        self.model = nnx.merge(graph_def, state)


def train_two_stage(
    model: nnx.Module,
    ref_model: nnx.Module,
    train_gsm8k_dataset: Any,
    train_nli_dataset: Any,
    grpo_config_stage1: Optional[GRPOConfig] = None,
    grpo_config_stage2: Optional[GRPOConfig] = None,
    checkpoint_dir: Optional[Path] = None,
    use_wandb: bool = False,
) -> Tuple[nnx.Module, Dict[str, Any]]:
    """Execute two-stage training pipeline.
    
    Stage 1: Train on GSM8K math reasoning
    Stage 2: Train on NLI with uncertainty awareness
    
    Args:
        model: Policy model
        ref_model: Reference model
        train_gsm8k_dataset: GSM8K training dataset
        train_nli_dataset: NLI training dataset
        grpo_config_stage1: GRPO config for stage 1
        grpo_config_stage2: GRPO config for stage 2
        checkpoint_dir: Directory for checkpoints
        use_wandb: Whether to use wandb logging
        
    Returns:
        Tuple of (trained model, metrics dict)
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
    
    all_metrics = {}
    
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
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
    )
    
    print("Starting Stage 1: GSM8K Training")
    metrics_stage1 = trainer_stage1.train(
        train_gsm8k_dataset,
        wandb_run_name="stage1_gsm8k",
    )
    all_metrics["stage1"] = metrics_stage1
    
    if checkpoint_dir:
        trainer_stage1.save_checkpoint(step=grpo_config_stage1.max_steps)
    
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
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
    )
    
    print("Starting Stage 2: NLI Training with Uncertainty Awareness")
    metrics_stage2 = trainer_stage2.train(
        train_nli_dataset,
        wandb_run_name="stage2_nli",
    )
    all_metrics["stage2"] = metrics_stage2
    
    if checkpoint_dir:
        trainer_stage2.save_checkpoint(step=grpo_config_stage2.max_steps)
    
    return trainer_stage2.model, all_metrics
