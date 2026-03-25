"""
Step 04a — MetaModel Module.

Defines:
- prepare_base_model() — validates fforward, extracts state structure
- MetaModel (nn.Module) — per-ticker meta-learning model
- MesaModel factory — inner optimizable model per ticker
- LinearCustom / LinearRestricted — custom nn.Linear variants
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base model preparation
# ---------------------------------------------------------------------------

def prepare_base_model(model: nn.Module, x: torch.Tensor) -> nn.Module:
    """
    Validate that model.fforward matches model.forward, then attach
    state helper functions and metadata to the model.
    """
    state = model.state_dict()

    model.eval()
    with torch.no_grad():
        output = model(x)
        output_ctrl = model.fforward(x, state)
    diff = (output - output_ctrl).abs().mean().item()
    if diff > 1e-10:
        raise RuntimeError(
            f"fforward does not match forward (mean diff = {diff:.2e})"
        )
    model.train()

    model.output_size = output.size(1)

    # --- State structure: ordered dict of param name → shape tuples ---
    model.state_structure = OrderedDict(
        {k: tuple(v.shape) for k, v in state.items()}
    )

    def flatten_state(sd: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.reshape(-1) for v in sd.values()])

    def unflatten_state(
        flat: torch.Tensor, structure: OrderedDict
    ) -> OrderedDict:
        sd = OrderedDict()
        offset = 0
        for name, shape in structure.items():
            numel = 1
            for s in shape:
                numel *= s
            sd[name] = flat[offset : offset + numel].view(shape)
            offset += numel
        return sd

    model.flatten_state = flatten_state
    model.unflatten_state = unflatten_state

    # Validate round-trip
    reconstructed = unflatten_state(flatten_state(state), model.state_structure)
    for k in state:
        if not torch.allclose(state[k], reconstructed[k]):
            raise RuntimeError("State flatten/unflatten round-trip failed.")

    model.state_size = int(flatten_state(state).numel())
    return model


# ---------------------------------------------------------------------------
# MetaModel
# ---------------------------------------------------------------------------

class MetaModel(nn.Module):
    """
    Per-ticker meta-learning model.

    Learns a *mesa parameter* per ticker (via xtype embedding), which is
    mapped through *meta weights* to produce per-ticker deviations from the
    base model's weights.
    """

    def __init__(
        self,
        base_model: nn.Module,
        xtype: torch.Tensor,
        mesa_parameter_size: int = 1,
        allow_bias: bool = True,
        p_dropout: float = 0.0,
        init_mesa_range: float = 0.0,
        init_meta_range: float = 1.0,
        allow_meta_structure: OrderedDict | None = None,
    ) -> None:
        super().__init__()

        self.fforward_fn = base_model.fforward
        self.state_structure = base_model.state_structure
        self.flatten_state = base_model.flatten_state
        self.unflatten_state = base_model.unflatten_state
        self.output_size = base_model.output_size
        self.state_size = base_model.state_size
        self.xtype_size = xtype.size(1)
        self.mesa_parameter_size = mesa_parameter_size
        self.allow_bias = allow_bias

        # Build allow-meta mask
        if allow_meta_structure is None:
            allow_mask = torch.ones(self.state_size, dtype=torch.bool)
        else:
            flat_parts = []
            for k, shape in base_model.state_structure.items():
                flat_parts.append(allow_meta_structure[k].reshape(-1).bool())
            allow_mask = torch.cat(flat_parts)
        self.register_buffer("allow_meta_vector", allow_mask)
        n_meta = int(allow_mask.sum().item())

        # Mesa layer: xtype → mesa parameter
        self.mesa_layer_weight = nn.Parameter(
            torch.empty(mesa_parameter_size, self.xtype_size).uniform_(
                -init_mesa_range, init_mesa_range
            )
        )
        # Meta layer: mesa parameter → weight deltas (only for allowed params)
        self.meta_layer_weight = nn.Parameter(
            torch.empty(n_meta, mesa_parameter_size).uniform_(
                -init_meta_range, init_meta_range
            )
        )
        if self.allow_bias:
            self.meta_layer_bias = nn.Parameter(
                torch.zeros(self.state_size)
            )

        # Store base state as a buffer (array to avoid cross-reference issues)
        base_state_flat = base_model.flatten_state(base_model.state_dict()).detach().cpu().numpy()
        self.register_buffer("base_state", torch.tensor(base_state_flat, dtype=torch.float32))

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        xtype: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        xout = torch.zeros(x.size(0), self.output_size)

        xtype_c = xtype.coalesce()
        columns = xtype_c.indices()[1]
        unique_cols = columns.unique().tolist()

        for i in unique_cols:
            mask = (columns == i)
            row_indices = xtype_c.indices()[0, mask]

            # One-hot for this ticker
            xtype_i = torch.zeros(self.xtype_size)
            xtype_i[i] = 1.0

            mesa_state = F.linear(xtype_i, self.mesa_layer_weight)

            temp = torch.zeros(self.state_size)
            temp[self.allow_meta_vector] = F.linear(mesa_state, self.meta_layer_weight)

            if self.allow_bias:
                meta_diff = self.meta_layer_bias + temp
            else:
                meta_diff = temp

            meta_state_flat = self.base_state + self.dropout(meta_diff)
            meta_state = self.unflatten_state(meta_state_flat, self.state_structure)

            xout[row_indices] = self.fforward_fn(x[row_indices], meta_state, **kwargs)

        return xout


# ---------------------------------------------------------------------------
# MesaModel factory (inner optimizable model per ticker)
# ---------------------------------------------------------------------------

def create_mesa_model(meta_model: MetaModel) -> type:
    """Return a nn.Module class that optimizes only the mesa parameter."""

    class MesaModelInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.fforward_fn = meta_model.fforward_fn
            self.state_structure = meta_model.state_structure
            self.unflatten_state = meta_model.unflatten_state
            self.state_size = meta_model.state_size
            self.allow_meta_vector = meta_model.allow_meta_vector
            self.allow_bias = meta_model.allow_bias

            sd = meta_model.state_dict()
            self.register_buffer("meta_layer_weight", sd["meta_layer_weight"].clone())
            if self.allow_bias:
                self.register_buffer("meta_layer_bias", sd["meta_layer_bias"].clone())
            self.register_buffer("base_state", meta_model.base_state.clone())

            self.mesa_state = nn.Parameter(
                torch.zeros(meta_model.mesa_parameter_size)
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            temp = torch.zeros(self.state_size)
            temp[self.allow_meta_vector] = F.linear(self.mesa_state, self.meta_layer_weight)
            if self.allow_bias:
                meta_diff = self.meta_layer_bias + temp
            else:
                meta_diff = temp
            meta_flat = self.base_state + meta_diff
            meta_state = self.unflatten_state(meta_flat, self.state_structure)
            return self.fforward_fn(x, meta_state, **kwargs)

    return MesaModelInner


# ---------------------------------------------------------------------------
# Custom linear layers
# ---------------------------------------------------------------------------

class LinearCustom(nn.Module):
    """nn.Linear with optional manual init and additive constant."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: np.ndarray | None = None,
        bias_init: float | None = None,
        init_range: float | None = None,
        const: float = 0.0,
    ) -> None:
        super().__init__()
        self.const = const
        if init_range is None:
            init_range = 1.0 / in_features ** 0.5
        if weight_init is None:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features).uniform_(-init_range, init_range)
            )
        else:
            self.weight = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
        if bias:
            if bias_init is None:
                self.bias = nn.Parameter(
                    torch.empty(out_features).uniform_(-init_range, init_range)
                )
            else:
                self.bias = nn.Parameter(torch.full((out_features,), bias_init))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + self.const


class LinearRestricted(nn.Module):
    """nn.Linear whose weight is a scalar times a fixed restriction vector."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        restriction: list[float],
        bias: bool = True,
        init_range: float | None = None,
    ) -> None:
        super().__init__()
        if init_range is None:
            init_range = 1.0 / in_features ** 0.5
        self.register_buffer(
            "restriction",
            torch.tensor(restriction, dtype=torch.float32).unsqueeze(0),
        )
        self.weight_scalar = nn.Parameter(
            torch.empty(1).uniform_(-init_range, init_range)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features).uniform_(-init_range, init_range)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_scalar * self.restriction
        return F.linear(x, weight, self.bias)
