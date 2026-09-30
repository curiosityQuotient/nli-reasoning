"""Tests for model loading and sharding configuration (CPU-only)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.models.loader import create_mesh, unshard_embedding_vocab

tunix_gemma = pytest.importorskip("tunix.models.gemma.model")


def _embedder_sharding_spec(shd_config, mesh):
    embedder = tunix_gemma.Embedder(64, 32, rngs=nnx.Rngs(0), shd_config=shd_config)
    return nnx.get_named_sharding(nnx.state(embedder), mesh)["input_embedding"].spec


def test_tunix_default_splits_embedding_vocab():
    """Document why the default is unusable: it shards the gathered axis."""
    default = tunix_gemma.ShardingConfig.get_default_sharding()
    spec = _embedder_sharding_spec(default, create_mesh())

    assert spec[0] is not None, "tunix changed its default embedding sharding"


def test_embedding_vocab_axis_is_replicated():
    mesh = create_mesh()
    default = tunix_gemma.ShardingConfig.get_default_sharding()
    spec = _embedder_sharding_spec(unshard_embedding_vocab(default), mesh)

    assert spec[0] is None, "token lookup gathers on dim 0; it must not be sharded"


def test_hidden_axis_keeps_its_original_mesh_axis():
    default = tunix_gemma.ShardingConfig.get_default_sharding()
    patched = unshard_embedding_vocab(default)

    assert patched.emb_vd[1:] == default.emb_vd[1:]
    assert patched.emb_vd[0] is None


def test_sharding_config_is_not_mutated():
    """The frozen input must be left alone; callers may reuse the default."""
    default = tunix_gemma.ShardingConfig.get_default_sharding()
    before = default.emb_vd

    unshard_embedding_vocab(default)

    assert default.emb_vd == before


def test_gather_over_embedding_is_not_sharding_ambiguous():
    """Regression: the gather that used to raise ShardingTypeError.

    Sharding a gather operand along the gathered axis leaves jax unable to
    pick an output sharding, which is what broke ``get_lora_model`` on a
    sharded mesh. Needs >1 device to be meaningful.
    """
    if len(jax.local_devices()) < 2:
        pytest.skip("needs 2+ devices to exercise a real mesh")

    mesh = jax.make_mesh((1, len(jax.local_devices())), ("fsdp", "tp"))
    from jax.sharding import NamedSharding, PartitionSpec

    table = jnp.zeros((256, 32), jnp.bfloat16)
    ids = jnp.zeros((2, 1, 1), jnp.int32)

    with jax.set_mesh(mesh):
        sharded = jax.device_put(table, NamedSharding(mesh, PartitionSpec("tp", None)))
        with pytest.raises(Exception, match="out_sharding"):
            jax.jit(lambda t, i: t[i])(sharded, ids)

        fixed = jax.device_put(table, NamedSharding(mesh, PartitionSpec(None, None)))
        assert jax.jit(lambda t, i: t[i])(fixed, ids).shape == (2, 1, 1, 32)
