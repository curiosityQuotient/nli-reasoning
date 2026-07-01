"""Model loading and initialization utilities."""

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import kagglehub
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh

from src.models.config import ALPHA, RANK, MESH


def download_gemma_model(
    model_family: str = "gemma2",
    model_version: str = "gemma2-2b-it"
) -> str:
    """Download Gemma model from Kaggle.
    
    Args:
        model_family: Model family name
        model_version: Model version string
        
    Returns:
        Path to downloaded model checkpoint
    """
    model_path = {
        "gemma2": "google/gemma-2/flax/",
    }
    
    kaggle_ckpt_path = kagglehub.model_download(
        f"{model_path[model_family]}{model_version}"
    )
    return kaggle_ckpt_path


def resave_checkpoint(
    kaggle_ckpt_path: str,
    output_path: str,
    model_family: str = "gemma2"
) -> None:
    """Resave model checkpoint in Flax NNX compatible format.
    
    Args:
        kaggle_ckpt_path: Path to Kaggle checkpoint
        output_path: Output path for resaved checkpoint
        model_family: Model family name
    """
    try:
        from tunix.models import gemma as gemma_lib
        from tunix.models import params as params_lib
    except ImportError:
        raise ImportError(
            "tunix package not installed. "
            "Install with: pip install google-tunix[prod]"
        )
    
    params = params_lib.load_and_format_params(kaggle_ckpt_path)
    
    if model_family == "gemma2":
        model = gemma_lib.Transformer.from_params(params)
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    checkpointer = ocp.StandardCheckpointer()
    state = nnx.state(model)
    checkpointer.save(output_path / "model_params", state)
    checkpointer.wait_until_finished()
    
    del params, model
    gc.collect()


def get_gemma_ref_model(
    ckpt_path: str,
    model_family: str = "gemma2",
    mesh: Optional[Mesh] = None
) -> nnx.Module:
    """Load Gemma reference model with JAX sharding.
    
    Args:
        ckpt_path: Path to model checkpoint
        model_family: Model family name
        mesh: JAX mesh for sharding
        
    Returns:
        Loaded model
    """
    try:
        from tunix.models import gemma as gemma_lib
    except ImportError:
        raise ImportError(
            "tunix package not installed. "
            "Install with: pip install google-tunix[prod]"
        )
    
    if mesh is None:
        devices = jax.local_devices()
        mesh = Mesh(
            jax.numpy.array(devices).reshape(1, len(devices)),
            ("fsdp", "tp")
        )
    
    with mesh:
        checkpointer = ocp.StandardCheckpointer()
        abs_state = ocp.args.StandardRestore(
            ckpt_path,
            restore_type=nnx.State
        )
        
        model = gemma_lib.Transformer()
        graph_def, abstract_state = nnx.split(model)
        state = checkpointer.restore(ckpt_path, abs_state)
        
        model = nnx.merge(graph_def, state)
        model = jax.jit(model)
    
    return model


def get_lora_model(
    base_model: nnx.Module,
    rank: int = RANK,
    alpha: float = ALPHA,
    mesh: Optional[Mesh] = None
) -> nnx.Module:
    """Apply LoRA to base model.
    
    Args:
        base_model: Base model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        mesh: JAX mesh for sharding
        
    Returns:
        Model with LoRA applied
    """
    try:
        import qwix
    except ImportError:
        raise ImportError(
            "qwix package not installed. "
            "Install with: pip install qwix"
        )
    
    if mesh is None:
        devices = jax.local_devices()
        mesh = Mesh(
            jax.numpy.array(devices).reshape(1, len(devices)),
            ("fsdp", "tp")
        )
    
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|"
            ".*down_proj|.*up_proj|.*attn_vec_einsum"
        ),
        rank=rank,
        alpha=alpha,
    )
    
    with mesh:
        model = lora_provider.wrap(base_model)
    
    return model


def get_tokenizer(
    model_version: str = "gemma2-2b-it",
    tokenizer_path: Optional[str] = None
) -> Any:
    """Get tokenizer for the model.
    
    Args:
        model_version: Model version string
        tokenizer_path: Optional path to tokenizer
        
    Returns:
        Tokenizer instance
    """
    try:
        from tunix.models import gemma as gemma_lib
    except ImportError:
        raise ImportError(
            "tunix package not installed. "
            "Install with: pip install google-tunix[prod]"
        )
    
    tokenizer = gemma_lib.Tokenizer(model_version)
    return tokenizer
