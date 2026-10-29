# License: https://github.com/yandex-research/tabm/blob/main/LICENSE
# Source: https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/nn_models/tabm.py
# Modifications for Mother: local embedding import, formatting, and type annotations.
# Apache-2.0 license text is distributed in tabm.LICENSE.

# NOTE
# The minimum required versions of the dependencies are specified in README.md.

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mother.ml.models import tabm_utils as rtdl_num_embeddings


# ======================================================================================
# Initialization
# ======================================================================================
def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    """Initialize a tensor uniformly using the inverse square root of ``d``.

    Args:
        x: Tensor to initialize in place.
        d: Positive fan-in used to calculate the uniform bound.

    Returns:
        The initialized tensor.
    """
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    """Initialize a tensor in place with randomly sampled ``-1`` and ``1``.

    Args:
        x: Tensor to initialize.

    Returns:
        The initialized tensor.
    """
    return x.bernoulli_(0.5).mul_(2).add_(-1)


# ======================================================================================
# Modules
# ======================================================================================
class NLinear(nn.Module):
    """N linear layers applied in parallel to N disjoint parts of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(self, n: int, in_features: int, out_features: int, bias: bool = True) -> None:
        """Create ``n`` independent linear transforms.

        Args:
            n: Number of parallel transforms.
            in_features: Input width of each transform.
            out_features: Output width of each transform.
            bias: Whether each transform has a bias.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases with inverse-square-root bounds."""
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each transform to its corresponding ensemble slice.

        Args:
            x: Input tensor with shape ``(batch, n, in_features)``.

        Returns:
            Tensor with shape ``(batch, n, out_features)``.
        """
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class OneHotEncoding0d(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: List[int]) -> None:
        """Configure one-hot encoding for each categorical feature.

        Args:
            cardinalities: Number of known categories in each feature. One
                additional index is reserved for unknown values and encoded
                as all zeros.
        """
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        """Encode categorical indices as concatenated one-hot vectors.

        Args:
            x: Integer category indices with the feature axis last.

        Returns:
            Concatenated one-hot features; unknown indices produce zeros.
        """
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


class ScaleEnsemble(nn.Module):
    def __init__(
        self,
        k: int,
        d: int,
        *,
        init: Literal["ones", "normal", "random-signs"],
    ) -> None:
        """Create a learned per-member scale over the flattened inputs.

        Args:
            k: Number of ensemble members.
            d: Number of input coordinates to scale.
            init: Initialization used for the scale weights.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, d))
        self._weight_init = init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the member scales according to ``self._weight_init``."""
        if self._weight_init == "ones":
            nn.init.ones_(self.weight)
        elif self._weight_init == "normal":
            nn.init.normal_(self.weight)
        elif self._weight_init == "random-signs":
            init_random_signs_(self.weight)
        else:
            raise ValueError(f"Unknown weight_init: {self._weight_init}")

    def forward(self, x: Tensor) -> Tensor:
        """Scale each member's input coordinates.

        Args:
            x: Input with ensemble members on the penultimate axis.

        Returns:
            The element-wise scaled input.
        """
        assert x.ndim >= 2
        return x * self.weight


class LinearEfficientEnsemble(nn.Module):
    """
    This layer is a more configurable version of the "BatchEnsemble" layer
    from the paper
    "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning"
    (link: https://arxiv.org/abs/2002.06715).

    First, this layer allows to select only some of the "ensembled" parts:
    - the input scaling  (r_i in the BatchEnsemble paper)
    - the output scaling (s_i in the BatchEnsemble paper)
    - the output bias    (not mentioned in the BatchEnsemble paper,
                          but is presented in public implementations)

    Second, the initialization of the scaling weights is configurable
    through the `scaling_init` argument.

    NOTE
    The term "adapter" is used in the TabM paper only to tell the story.
    The original BatchEnsemble paper does NOT use this term. So this class also
    avoids the term "adapter".
    """

    r: Optional[Tensor]
    s: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal["ones", "random-signs"],
    ) -> None:
        """Configure the shared linear transform and member-specific factors.

        Args:
            in_features: Input width of the shared linear transform.
            out_features: Output width of the shared linear transform.
            bias: Whether the transform includes a bias.
            k: Number of ensemble members.
            ensemble_scaling_in: Learn an input scale for each member.
            ensemble_scaling_out: Learn an output scale for each member.
            ensemble_bias: Learn a separate bias for each member.
            scaling_init: Initialization for member-specific scale factors.
        """
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            "r",
            (nn.Parameter(torch.empty(k, in_features)) if ensemble_scaling_in else None),  # type: ignore[code]
        )
        self.register_parameter(
            "s",
            (nn.Parameter(torch.empty(k, out_features)) if ensemble_scaling_out else None),  # type: ignore[code]
        )
        self.register_parameter(
            "bias",
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize shared weights, member scales, and biases."""
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {"ones": nn.init.ones_, "random-signs": init_random_signs_}[self.scaling_init]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the shared linear transform with member-specific factors.

        Args:
            x: Input tensor with shape ``(batch, members, in_features)``.

        Returns:
            Tensor with shape ``(batch, members, out_features)``.
        """
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class MLP(nn.Module):
    """Feed-forward MLP backbone used by the TabM ensemble."""

    def __init__(
        self,
        *,
        d_in: Optional[int] = None,
        d_out: Optional[int] = None,
        n_blocks: int,
        d_block: int,
        dropout: float,
        activation: str = "ReLU",
    ) -> None:
        """Build an MLP with configurable depth, width, and activation.

        Args:
            d_in: Input width; defaults to ``d_block`` when omitted.
            d_out: Optional output width for a final linear layer.
            n_blocks: Number of hidden blocks.
            d_block: Width of each hidden block.
            dropout: Dropout probability applied after each activation.
            activation: Name of the PyTorch activation class in ``torch.nn``.
        """
        super().__init__()

        d_first = d_block if d_in is None else d_in
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_first if i == 0 else d_block, d_block),
                    getattr(nn, activation)(),
                    nn.Dropout(dropout),
                )
                for i in range(n_blocks)
            ]
        )
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the hidden blocks and optional output projection.

        Args:
            x: Input tensor whose final dimension matches the configured
                input width.

        Returns:
            The transformed tensor.
        """
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


def make_efficient_ensemble(module: nn.Module, EnsembleLayer: Callable[..., nn.Module], **kwargs: Any) -> None:
    """Replace linear layers with efficient ensembles of linear layers.

    Args:
        module: Module tree whose linear layers should be replaced in place.
        EnsembleLayer: Factory used to create each replacement layer.
        **kwargs: Ensemble-specific keyword arguments forwarded to the factory.

    NOTE
    In the paper, there are no experiments with networks with normalization layers.
    Perhaps, their trainable weights (the affine transformations) also need
    "ensemblification" as in the paper about "FiLM-Ensemble".
    Additional experiments are required to make conclusions.
    """
    for name, submodule in list(module.named_children()):
        if isinstance(submodule, nn.Linear):
            module.add_module(
                name,
                EnsembleLayer(
                    in_features=submodule.in_features,
                    out_features=submodule.out_features,
                    bias=submodule.bias is not None,
                    **kwargs,
                ),
            )
        else:
            make_efficient_ensemble(submodule, EnsembleLayer, **kwargs)


def _get_first_ensemble_layer(backbone: MLP) -> LinearEfficientEnsemble:
    """Return the first ensemble layer in the MLP backbone.

    Args:
        backbone: MLP whose first block has already been ensembled.

    Returns:
        The first block's linear ensemble layer.
    """
    if isinstance(backbone, MLP):
        return backbone.blocks[0][0]  # type: ignore[code]
    else:
        raise RuntimeError(f"Unsupported backbone: {backbone}")


@torch.inference_mode()
def _init_first_adapter(
    weight: Tensor,
    distribution: Literal["normal", "random-signs"],
    init_sections: List[int],
) -> None:
    """Initialize the first adapter.

    NOTE
    The `init_sections` argument is a historical artifact that accidentally leaked
    from irrelevant experiments to the final models. Perhaps, the code related
    to `init_sections` can be simply removed, but this was not tested.
    """
    assert weight.ndim == 2
    assert weight.shape[1] == sum(init_sections)

    if distribution == "normal":
        init_fn_ = nn.init.normal_
    elif distribution == "random-signs":
        init_fn_ = init_random_signs_
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        # NOTE
        # As noted above, this section-based initialization is an arbitrary historical
        # artifact. Consider the first adapter of one ensemble member.
        # This adapter vector is implicitly split into "sections",
        # where one section corresponds to one feature. The code below ensures that
        # the adapter weights in one section are initialized with the same random value
        # from the given distribution.
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w


_CUSTOM_MODULES: Dict[str, type[nn.Module]] = {
    # https://docs.python.org/3/library/stdtypes.html#definition.__name__
    CustomModule.__name__: CustomModule
    for CustomModule in [
        rtdl_num_embeddings.LinearEmbeddings,
        rtdl_num_embeddings.LinearReLUEmbeddings,
        rtdl_num_embeddings.PeriodicEmbeddings,
        rtdl_num_embeddings.PiecewiseLinearEmbeddings,
        MLP,
    ]
}


def make_module(type: str, *args: Any, **kwargs: Any) -> nn.Module:
    """Construct a PyTorch or registered custom module by name.

    Args:
        type: Module class name, searched in ``torch.nn`` before the custom
            module registry.
        *args: Positional constructor arguments.
        **kwargs: Keyword constructor arguments.

    Returns:
        An initialized PyTorch module.
    """
    Module = getattr(nn, type, None)
    if Module is None:
        Module = _CUSTOM_MODULES[type]
    return Module(*args, **kwargs)


# ======================================================================================
# Optimization
# ======================================================================================
def default_zero_weight_decay_condition(
    module_name: str, module: nn.Module, parameter_name: str, parameter: nn.Parameter
) -> bool:
    """Return whether a parameter should be excluded from weight decay.

    Biases and parameters belonging to normalization or numerical embedding
    modules are excluded.

    Args:
        module_name: Name of the module containing the parameter.
        module: Module that owns the parameter.
        parameter_name: Local parameter name.
        parameter: Parameter being checked.

    Returns:
        Whether the parameter belongs in the zero-weight-decay group.
    """

    del module_name, parameter
    return parameter_name.endswith("bias") or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            rtdl_num_embeddings.LinearEmbeddings,
            rtdl_num_embeddings.LinearReLUEmbeddings,
            rtdl_num_embeddings._Periodic,
        ),
    )


def make_parameter_groups(
    module: nn.Module,
    zero_weight_decay_condition: Callable[
        [str, nn.Module, str, nn.Parameter], bool
    ] = default_zero_weight_decay_condition,
    custom_groups: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Partition module parameters into weight-decay optimizer groups.

    Args:
        module: PyTorch module whose parameters are grouped.
        zero_weight_decay_condition: Predicate selecting parameters that
            receive no weight decay.
        custom_groups: Additional optimizer groups, which must not overlap.

    Returns:
        Optimizer parameter groups: regular parameters, zero-decay
        parameters, then any caller-provided groups.
    """
    if custom_groups is None:
        custom_groups = []
    custom_params = frozenset(itertools.chain.from_iterable(group["params"] for group in custom_groups))
    assert len(custom_params) == sum(len(group["params"]) for group in custom_groups), (
        "Parameters in custom_groups must not intersect"
    )
    zero_wd_params = frozenset(
        p
        for mn, m in module.named_modules()
        for pn, p in m.named_parameters()
        if p not in custom_params and zero_weight_decay_condition(mn, m, pn, p)
    )
    default_group = {"params": [p for p in module.parameters() if p not in custom_params and p not in zero_wd_params]}
    return [
        default_group,
        {"params": list(zero_wd_params), "weight_decay": 0.0},
        *custom_groups,
    ]


# ======================================================================================
# The model
# ======================================================================================
class Model(nn.Module):
    """TabM ensemble around an MLP backbone.

    Numeric and categorical inputs are transformed separately and combined
    before the backbone. Ensemble adapters share most backbone parameters
    while producing one output per member.
    """

    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        backbone: Dict[str, Any],
        bins: Optional[List[Tensor]],  # For piecewise-linear encoding/embeddings.
        num_embeddings: Optional[Dict[str, Any]] = None,
        arch_type: Literal[
            # Plain feed-forward network without any kind of ensembling.
            "plain",
            #
            # TabM
            "tabm",
            #
            # TabM-mini
            "tabm-mini",
            #
            # TabM-packed
            "tabm-packed",
            #
            # TabM. The first adapter is initialized from the normal distribution.
            # This variant was not used in the paper, but it may be useful in practice.
            "tabm-normal",
            #
            # TabM-mini. The adapter is initialized from the normal distribution.
            # This variant was not used in the paper.
            "tabm-mini-normal",
        ],
        k: Optional[int] = None,
        share_training_batches: bool = True,
    ) -> None:
        """Configure input transforms, the MLP, ensemble adapters, and output.

        Args:
            n_num_features: Number of numeric input columns.
            cat_cardinalities: Number of known categories per categorical
                column.
            n_classes: Number of classes, or ``None`` for regression output.
            backbone: Constructor settings for the backbone module.
            bins: Training-derived bin edges for piecewise-linear embeddings.
            num_embeddings: Optional numerical embedding module settings.
            arch_type: Network architecture variant.
            k: Number of ensemble members; omitted only for a plain network.
            share_training_batches: Whether all members share each input
                batch during training.
        """
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == "plain":
            assert k is None
            assert share_training_batches, 'If `arch_type` is set to "plain", then `simple` must remain True'
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        # >>> Continuous (numerical) features
        first_adapter_sections: List[int] = []  # See the comment in `_init_first_adapter`.

        if n_num_features == 0:
            assert bins is None
            self.num_module = None
            d_num = 0

        elif num_embeddings is None:
            assert bins is None
            self.num_module = None
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        else:
            if bins is None:
                self.num_module = make_module(**num_embeddings, n_features=n_num_features)
            else:
                assert num_embeddings["type"].startswith("PiecewiseLinearEmbeddings")
                self.num_module = make_module(**num_embeddings, bins=bins)
            d_num = n_num_features * num_embeddings["d_embedding"]
            first_adapter_sections.extend(num_embeddings["d_embedding"] for _ in range(n_num_features))

        # >>> Categorical features
        self.cat_module = OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        first_adapter_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None
        # Any backbone can be here but we provide only MLP
        self.backbone = make_module(d_in=d_flat, **backbone)

        if arch_type != "plain":
            assert k is not None
            first_adapter_init = (
                None
                if arch_type == "tabm-packed"
                else "normal"
                if arch_type in ("tabm-mini-normal", "tabm-normal")
                # For other arch_types, the initialization depends
                # on the presence of num_embeddings.
                else "random-signs"
                if num_embeddings is None
                else "normal"
            )

            if arch_type in ("tabm", "tabm-normal"):
                # Like BatchEnsemble, but all multiplicative adapters,
                # except for the very first one, are initialized with ones.
                assert first_adapter_init is not None
                make_efficient_ensemble(
                    self.backbone,
                    LinearEfficientEnsemble,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init="ones",
                )
                _init_first_adapter(
                    _get_first_ensemble_layer(self.backbone).r,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type in ("tabm-mini", "tabm-mini-normal"):
                # MiniEnsemble
                assert first_adapter_init is not None
                self.minimal_ensemble_adapter = ScaleEnsemble(
                    k,
                    d_flat,
                    init="random-signs" if num_embeddings is None else "normal",
                )
                _init_first_adapter(
                    self.minimal_ensemble_adapter.weight,  # type: ignore[code]
                    first_adapter_init,
                    first_adapter_sections,
                )

            elif arch_type == "tabm-packed":
                # Packed ensemble.
                # In terms of the Packed Ensembles paper by Laurent et al.,
                # TabM-packed is PackedEnsemble(alpha=k, M=k, gamma=1).
                assert first_adapter_init is None
                make_efficient_ensemble(self.backbone, NLinear, n=k)

            else:
                raise ValueError(f"Unknown arch_type: {arch_type}")

        # >>> Output
        d_block = backbone["d_block"]
        d_out = 1 if n_classes is None else n_classes
        if arch_type == "plain":
            self.output = nn.Linear(d_block, d_out)
        else:
            assert k is not None
            self.output = NLinear(k, d_block, d_out)

        # >>>
        self.arch_type = arch_type
        self.k = k
        self.share_training_batches = share_training_batches

    def forward(self, x_num: Optional[Tensor] = None, x_cat: Optional[Tensor] = None) -> Tensor:
        """Compute predictions for each ensemble member.

        Args:
            x_num: Optional numeric input with shape ``(batch, features)``.
            x_cat: Optional integer categorical input with shape
                ``(batch, categorical_features)``.

        Returns:
            Tensor shaped ``(batch, members, outputs)``. A plain network uses
            a singleton member dimension for the same output convention.
        """
        x = []
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat).float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.k is not None:
            if self.share_training_batches or not self.training:
                # (B, D) -> (B, K, D)
                x = x[:, None].expand(-1, self.k, -1)
            else:
                # (B * K, D) -> (B, K, D)
                x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)
        else:
            assert self.minimal_ensemble_adapter is None

        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            # Adjust the output shape for plain networks to make them compatible
            # with the rest of the script (loss, metrics, predictions, ...).
            # (B, D_OUT) -> (B, 1, D_OUT)
            x = x[:, None]
        return x
