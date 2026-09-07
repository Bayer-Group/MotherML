"""NODE head modules used by m_node.

Contains only the head implementations required by NODE and helper functions
used in NODE flow prediction paths.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

try:
    import zuko
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    zuko = None  # type: ignore[assignment]


class MLPHead(nn.Module):
    """MLP readout head used by NODE."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "ReLU",
        norm: str = "batch",
    ) -> None:
        """Build a stack of Linear/Norm/activation/Dropout blocks from `input_dim` to `output_dim`."""
        super().__init__()

        def _make_activation() -> nn.Module:
            """Instantiate the configured activation module by name."""
            if activation == "ReLU":
                return nn.ReLU()
            if activation == "GELU":
                return nn.GELU()
            if activation == "LeakyReLU":
                return nn.LeakyReLU()
            if activation == "ELU":
                return nn.ELU()
            if activation == "SiLU":
                return nn.SiLU()
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm != "none":
                    raise ValueError(f"Unsupported norm: {norm!r}. Choose 'batch', 'layer' or 'none'.")
                layers.append(_make_activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """Flatten the input (or concatenate tensor `**kwargs`) and run it through the MLP."""
        if x is None:
            if not kwargs:
                raise ValueError("No input data provided to forward()")
            tensors = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise ValueError("No input data provided to forward()")
            tensors_2d = [t.view(-1, 1) if t.dim() == 1 else t for t in tensors]
            x = torch.cat(tensors_2d, dim=1)

        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.mlp(x)


class FlowHead(nn.Module):
    """Conditional flow readout head used by NODE."""

    SUPPORTED_FLOW_TYPES = ("GMM", "NICE", "RealNVP", "NAF", "UNAF", "NSF", "BPF")

    @staticmethod
    def _move_nested_tensors_to_device(obj: Any, device: torch.device, visited: Optional[set[int]] = None) -> Any:
        """Recursively move any tensors found inside `obj` (dict/list/tuple/object graph) to `device` in place."""
        if visited is None:
            visited = set()

        oid = id(obj)
        if oid in visited:
            return obj
        visited.add(oid)

        if isinstance(obj, torch.Tensor):
            if obj.device != device:
                return obj.to(device)
            return obj

        if isinstance(obj, nn.Parameter):
            return obj

        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                obj[k] = FlowHead._move_nested_tensors_to_device(v, device, visited)
            return obj

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                obj[i] = FlowHead._move_nested_tensors_to_device(v, device, visited)
            return obj

        if isinstance(obj, tuple):
            return tuple(FlowHead._move_nested_tensors_to_device(v, device, visited) for v in obj)

        if hasattr(obj, "__dict__"):
            for attr_name, attr_val in list(vars(obj).items()):
                try:
                    new_val = FlowHead._move_nested_tensors_to_device(attr_val, device, visited)
                    if new_val is not attr_val:
                        setattr(obj, attr_name, new_val)
                except Exception:
                    continue
        return obj

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        flow_type: str = "NSF",
        flow_transforms: int = 3,
        flow_bins: int = 8,
        flow_degree: int = 16,
        flow_signal: int = 16,
        flow_components: int = 8,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_dropout: float = 0.0,
        mlp_activation: str = "GELU",
        mlp_norm: str = "batch",
    ) -> None:
        """Build a zuko conditional flow of type `flow_type`, with an optional MLP conditioner encoder."""
        super().__init__()
        if zuko is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "zuko is required for FlowHead. Install optional dependencies, e.g. `pip install mother-ml[node]`."
            )

        self.mlp_hidden_dims = list(mlp_hidden_dims) if mlp_hidden_dims else None

        if self.mlp_hidden_dims:

            def _make_activation() -> nn.Module:
                """Instantiate the configured MLP-conditioner activation module by name."""
                if mlp_activation == "ReLU":
                    return nn.ReLU()
                if mlp_activation == "GELU":
                    return nn.GELU()
                if mlp_activation == "LeakyReLU":
                    return nn.LeakyReLU()
                if mlp_activation == "ELU":
                    return nn.ELU()
                if mlp_activation == "SiLU":
                    return nn.SiLU()
                if mlp_activation == "Tanh":
                    return nn.Tanh()
                raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")

            encoder_layers: List[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in self.mlp_hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                if mlp_norm == "batch":
                    encoder_layers.append(nn.BatchNorm1d(hidden_dim))
                elif mlp_norm == "layer":
                    encoder_layers.append(nn.LayerNorm(hidden_dim))
                elif mlp_norm != "none":
                    raise ValueError(f"Unsupported mlp_norm: {mlp_norm!r}. Choose 'batch', 'layer' or 'none'.")
                encoder_layers.append(_make_activation())
                if mlp_dropout > 0:
                    encoder_layers.append(nn.Dropout(mlp_dropout))
                prev_dim = hidden_dim

            self.encoder: Optional[nn.Module] = nn.Sequential(*encoder_layers)
            for module in self.encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            context_dim = self.mlp_hidden_dims[-1]
        else:
            self.encoder = None
            context_dim = input_dim

        if flow_type == "GMM":
            self.net = zuko.flows.GMM(features=output_dim, context=context_dim, components=flow_components)
        elif flow_type == "NICE":
            self.net = zuko.flows.NICE(features=output_dim, context=context_dim, transforms=flow_transforms)
        elif flow_type == "RealNVP":
            self.net = zuko.flows.RealNVP(features=output_dim, context=context_dim, transforms=flow_transforms)
        elif flow_type == "NAF":
            self.net = zuko.flows.NAF(
                features=output_dim, context=context_dim, transforms=flow_transforms, signal=flow_signal
            )
        elif flow_type == "UNAF":
            self.net = zuko.flows.UNAF(
                features=output_dim, context=context_dim, transforms=flow_transforms, signal=flow_signal
            )
        elif flow_type == "NSF":
            self.net = zuko.flows.NSF(features=output_dim, context=context_dim, bins=flow_bins)
        elif flow_type == "BPF":
            self.net = zuko.flows.BPF(features=output_dim, context=context_dim, degree=flow_degree)
        else:
            raise ValueError(f"Unsupported flow_type: {flow_type}. Choose from {self.SUPPORTED_FLOW_TYPES}.")

    def forward(self, x: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        """
        Build a conditional predictive distribution from node embeddings.

        Args:
            x: Node embeddings, shape ``[batch_size, input_dim]``. If ``None``,
                built from `**kwargs` tensor values instead (concatenated column-wise).
            **kwargs: Alternative tensor inputs used when `x` is not provided.

        Returns:
            A zuko distribution object conditioned on `x`, exposing `.sample()`,
            `.rsample()` and `.log_prob()` for downstream sampling and density evaluation.
        """
        if x is None:
            if not kwargs:
                raise ValueError("No input data provided to forward()")
            tensors = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise ValueError("No input data provided to forward()")
            tensors_2d = [t.view(-1, 1) if t.dim() == 1 else t for t in tensors]
            x = torch.cat(tensors_2d, dim=1)

        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        if self.encoder is not None:
            x = self.encoder(x)

        dist = self.net(x)
        self._move_nested_tensors_to_device(dist, x.device)
        return dist


def compute_flow_mode_and_uncertainty(dist: Any, num_samples: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow mode predictions and data uncertainty from sampled log-prob.

    Args:
        dist: Distribution object with `sample()` and `log_prob()`.
        num_samples: Number of samples drawn per input to approximate the mode.

    Returns:
        Tuple[Tensor, Tensor]:
        - mode predictions with shape [batch_size, output_dim]
        - uncertainty values with shape [batch_size] as -log_prob(mode)
    """
    with torch.no_grad():
        samples = dist.sample((num_samples,))
        log_probs = dist.log_prob(samples)
        best_log_probs, best_indices = log_probs.max(dim=0)
        batch_arange = torch.arange(samples.shape[1], device=samples.device)
        mode_predictions = samples[best_indices, batch_arange, :]
        uncertainties = -best_log_probs

    return mode_predictions, uncertainties
