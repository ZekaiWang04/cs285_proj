import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Union
import pathlib

def save_leaves(
    model: eqx.Module, path: Union[str, pathlib.Path], filter_spec=eqx.is_array_like
):
    # usage: save_leaves(model, path)
    tree_weights, _ = eqx.partition(model, filter_spec)

    with open(path, "wb") as f:
        for x in jax.tree_util.tree_leaves(tree_weights):
            jnp.save(f, x, allow_pickle=False)


def load_leaves(
    model: eqx.Module, path: Union[str, pathlib.Path], filter_spec=eqx.is_array_like
):
    # usage: new_model = SomeClass(...) # must be same as model stored in path
    #        new_model = load_leaves(new_model, path)
    tree_weights, tree_other = eqx.partition(model, filter_spec)

    leaves_orig, treedef = jax.tree_util.tree_flatten(tree_weights)
    with open(path, "rb") as f:
        flat_state = [jnp.asarray(jnp.load(f)) for _ in leaves_orig]
    tree_weights = jax.tree_util.tree_unflatten(treedef, flat_state)
    return eqx.combine(tree_weights, tree_other)