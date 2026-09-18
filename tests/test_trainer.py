"""Integration tests for the GRPO trainer configuration (CPU-only)."""

from src.models.config import TOTAL_GENERATION_STEPS, TRAIN_MICRO_BATCH_SIZE
from src.models.loader import create_mesh
from src.training.grpo_config import GRPOConfig
from src.training.trainer import GRPOTrainer, create_optimizer


def _make_trainer(checkpoint_dir=None) -> GRPOTrainer:
    return GRPOTrainer(
        model=object(),
        ref_model=object(),
        optimizer=create_optimizer(),
        grpo_config=GRPOConfig(),
        reward_fns=[lambda **kwargs: None],
        tokenizer=object(),
        mesh=create_mesh(),
        checkpoint_dir=checkpoint_dir,
    )


def test_create_optimizer_returns_transformation():
    optimizer = create_optimizer()
    assert optimizer is not None


def test_cluster_config_matches_tunix_api():
    """The built ClusterConfig must satisfy the Tunix 0.1.7 schema."""
    trainer = _make_trainer()
    config = trainer._build_cluster_config(total_steps=5)

    from tunix.rl.rl_cluster import Role

    assert set(config.role_to_mesh) == {Role.ACTOR, Role.REFERENCE, Role.ROLLOUT}
    assert config.rollout_engine == "vanilla"
    assert config.training_config.max_steps == 5
    assert config.training_config.actor_optimizer is trainer.optimizer
    assert config.training_config.train_micro_batch_size == TRAIN_MICRO_BATCH_SIZE
    assert config.rollout_config.max_tokens_to_generate == TOTAL_GENERATION_STEPS
    assert (
        config.rollout_config.max_prompt_length <= config.rollout_config.kv_cache_size
    )


def test_checkpoint_config_uses_stage_directory(tmp_path):
    """Checkpoint roots are stage-specific and saved at the final step."""
    trainer = _make_trainer(checkpoint_dir=tmp_path / "stage1")
    config = trainer._build_cluster_config(total_steps=7)

    assert config.training_config.checkpoint_root_directory == str(tmp_path / "stage1")
    assert config.training_config.checkpointing_options.save_interval_steps == 7


def test_no_checkpoint_by_default():
    config = _make_trainer()._build_cluster_config(total_steps=5)
    assert config.training_config.checkpoint_root_directory is None
    assert config.training_config.checkpointing_options is None


def test_stage_configs_differ_in_iterations():
    from src.training.grpo_config import (
        create_grpo_config_stage1,
        create_grpo_config_stage2,
    )

    stage1 = create_grpo_config_stage1()
    stage2 = create_grpo_config_stage2()
    assert stage2.num_iterations > stage1.num_iterations
