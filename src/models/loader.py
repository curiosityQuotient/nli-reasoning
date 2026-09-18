"""Model loading and initialization utilities.

These helpers mirror the notebook's model setup: download the Kaggle
checkpoint, resave it in a Flax NNX compatible format, restore a sharded
reference model, apply LoRA to a fresh instance, and set up the tokenizer.
"""

import gc
from pathlib import Path
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import kagglehub
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh

from src.models.config import ALPHA, MESH_AXES, MESH_SHAPE, RANK

GEMMA2_FROM_PARAMS_VERSION = "2-2b-it"


def download_gemma_model(
    model_family: str = "gemma2", model_version: str = "gemma2-2b-it"
) -> str:
    """Download Gemma model from Kaggle."""
    model_path = {
        "gemma2": "google/gemma-2/flax/",
    }
    if model_family not in model_path:
        raise ValueError(f"Unknown model family: {model_family}")

    return kagglehub.model_download(f"{model_path[model_family]}{model_version}")


def _resolve_checkpoint_dir(ckpt_path: Path) -> Path:
    """Find the checkpoint directory containing Orbax metadata.

    KaggleHub downloads may nest the actual checkpoint inside numbered
    subdirectories; descend until the ``_METADATA`` marker is found.
    """
    if (ckpt_path / "_METADATA").exists():
        return ckpt_path

    subdirs = [p for p in ckpt_path.iterdir() if p.is_dir()]
    for subdir in subdirs:
        if (subdir / "_METADATA").exists():
            return subdir
    if len(subdirs) == 1:
        return _resolve_checkpoint_dir(subdirs[0])
    raise FileNotFoundError(f"No Orbax checkpoint found under: {ckpt_path}")


def _from_params_version(model_family: str, model_version: str) -> str:
    """Map a Kaggle model version to Tunix's ``from_params`` version id.

    Example: ("gemma2", "gemma2-2b-it") -> "2-2b-it", which resolves to
    ``ModelConfig.gemma2_2b_it``.
    """
    if model_family != "gemma2":
        raise ValueError(f"Unsupported model family: {model_family}")
    stripped = model_version.lower().replace("gemma2-", "").strip("-_")
    return f"2-{stripped}"


def resave_checkpoint(
    kaggle_ckpt_path: str,
    output_path: str,
    model_family: str = "gemma2",
    model_version: str = "gemma2-2b-it",
) -> None:
    """Resave model checkpoint in Flax NNX compatible format."""
    try:
        from tunix.models.gemma import model as gemma_model
        from tunix.models.gemma import params as params_lib
    except ImportError as e:
        raise ImportError(
            f"Failed to import tunix modules: {e}. "
            "Ensure google-tunix is installed with: pip install google-tunix"
        ) from e

    ckpt_path = _resolve_checkpoint_dir(Path(kaggle_ckpt_path))
    params = params_lib.load_and_format_params(str(ckpt_path))
    gemma = gemma_model.Gemma.from_params(
        params,
        version=_from_params_version(model_family, model_version),
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma)
    checkpointer.save(output_path / "model_params", state)
    checkpointer.wait_until_finished()

    del params, gemma, state
    gc.collect()


def create_mesh() -> Mesh:
    """Create the FSDP/TP mesh described by the config.

    Falls back to a single-device mesh (still exposing both axes, sized
    1) when fewer devices are available than requested.
    """
    devices = jax.local_devices()
    shape = MESH_SHAPE
    if len(devices) < MESH_SHAPE[0] * MESH_SHAPE[1]:
        shape = (1, len(devices))
    return jax.make_mesh(shape, MESH_AXES)


def get_gemma_ref_model(
    ckpt_path: str,
) -> Tuple[nnx.Module, Mesh, Any]:
    """Load a sharded Gemma reference model from an NNX checkpoint.

    Args:
        ckpt_path: Path to the resaved NNX checkpoint directory

    Returns:
        Tuple of (model, mesh, model_config)
    """
    try:
        from tunix.models.gemma import model as gemma_model
    except ImportError as e:
        raise ImportError(
            f"Failed to import tunix modules: {e}. "
            "Install with: pip install google-tunix"
        ) from e

    mesh = create_mesh()
    model_config = gemma_model.ModelConfig.gemma2_2b()

    abs_gemma: nnx.Module = nnx.eval_shape(
        lambda: gemma_model.Gemma(model_config, rngs=nnx.Rngs(params=0))
    )
    abs_state = nnx.state(abs_gemma)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )

    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    graph_def, _ = nnx.split(abs_gemma)
    model = nnx.merge(graph_def, restored_params)
    return model, mesh, model_config


def get_lora_model(
    base_model: nnx.Module,
    mesh: Mesh,
    rank: int = RANK,
    alpha: float = ALPHA,
) -> nnx.Module:
    """Apply LoRA to a base model, returning a fresh policy instance.

    The base model object is left untouched and remains usable as the
    frozen reference model.

    Args:
        base_model: Reference model (plain NNX module)
        mesh: JAX mesh used to shard the resulting model
        rank: LoRA rank
        alpha: LoRA alpha scaling

    Returns:
        LoRA-adapted model
    """
    try:
        import qwix
    except ImportError as e:
        raise ImportError(
            "qwix package not installed. Install with: pip install qwix"
        ) from e

    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"
        ),
        rank=rank,
        alpha=alpha,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model,
        lora_provider,
        **model_input,
        rngs=nnx.Rngs(0),
    )

    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


def get_tokenizer(tokenizer_path: Optional[str] = None) -> Any:
    """Get the Gemma tokenizer.

    Args:
        tokenizer_path: SentencePiece model path; defaults to the public
            Gemma2 tokenizer hosted on GCS

    Returns:
        Tokenizer instance
    """
    try:
        from tunix.generate import tokenizer_adapter as tokenizer_lib
    except ImportError as e:
        raise ImportError(
            f"Failed to import tunix modules: {e}. "
            "Install with: pip install google-tunix"
        ) from e

    return tokenizer_lib.Tokenizer(
        tokenizer_path=(
            tokenizer_path or "gs://gemma-data/tokenizers/tokenizer_gemma2.model"
        )
    )
