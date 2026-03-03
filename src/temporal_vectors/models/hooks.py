"""Forward-hook utilities for activation steering and hidden-state capture."""

from contextlib import contextmanager
from typing import Callable

import torch
from torch import nn


def make_steering_hook(
    direction: torch.Tensor,
    alpha: float = 1.0,
    token_position: int = -1,
) -> Callable:
    """Create a forward hook that adds a scaled direction to hidden states.

    Args:
        direction: Steering vector, shape (hidden_size,).
        alpha: Scaling factor for the steering direction.
        token_position: Which token position to steer (-1 = last token only,
            None = all tokens).

    Returns:
        A hook function compatible with register_forward_hook.
    """

    def hook(module: nn.Module, input: tuple, output: tuple) -> tuple:
        hidden_states = output[0]
        steering = alpha * direction.to(hidden_states.device, dtype=hidden_states.dtype)
        if token_position is None:
            hidden_states = hidden_states + steering
        else:
            hidden_states[:, token_position, :] = (
                hidden_states[:, token_position, :] + steering
            )
        return (hidden_states,) + output[1:]

    return hook


def make_capture_hook(storage: dict, layer_id: int) -> Callable:
    """Create a forward hook that captures hidden states into a dict.

    Args:
        storage: Dict to store captured tensors. After the forward pass,
            storage[layer_id] will contain the hidden states tensor.
        layer_id: Key to use in the storage dict.

    Returns:
        A hook function compatible with register_forward_hook.
    """

    def hook(module: nn.Module, input: tuple, output: tuple) -> None:
        storage[layer_id] = output[0].detach().cpu().float()

    return hook


@contextmanager
def steer_model(
    model: nn.Module,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float = 1.0,
    token_position: int = -1,
):
    """Context manager that temporarily steers a model layer.

    Usage:
        with steer_model(model, layer_idx=14, direction=gamma, alpha=1.0):
            output = model.generate(...)

    Args:
        model: The causal LM (expects model.model.layers).
        layer_idx: Which transformer layer to hook.
        direction: Steering vector, shape (hidden_size,).
        alpha: Scaling factor.
        token_position: Which token position to steer.
    """
    hook_fn = make_steering_hook(direction, alpha, token_position)
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    try:
        yield handle
    finally:
        handle.remove()


@contextmanager
def capture_hidden_states(
    model: nn.Module,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Context manager that captures hidden states from specified layers.

    Usage:
        storage = {}
        with capture_hidden_states(model, [7, 14, 21, 27]) as captured:
            model(**inputs)
        # captured[7], captured[14], etc. now hold the hidden states

    Args:
        model: The causal LM (expects model.model.layers).
        layer_indices: Which transformer layers to capture.

    Yields:
        Dict mapping layer_idx -> captured hidden states tensor.
    """
    storage: dict[int, torch.Tensor] = {}
    handles = []
    for idx in layer_indices:
        hook_fn = make_capture_hook(storage, idx)
        handle = model.model.layers[idx].register_forward_hook(hook_fn)
        handles.append(handle)
    try:
        yield storage
    finally:
        for handle in handles:
            handle.remove()
