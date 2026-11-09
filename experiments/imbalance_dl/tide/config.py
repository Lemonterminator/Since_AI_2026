"""Configuration for the PyTorch TiDE port and the imbalance task bridge.

``known_channels`` and ``observed_channels`` are integer channel counts.  They
describe the ``ForecastBatch`` tensors, not the augmented tensors that the task
bridge builds for the native model.  The task bridge expands every known
channel with its mask and age, so the native feature projection sees
``3 * known_channels`` values.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TiDEConfig:
    """Electricity-paper settings with the competition's forty-step horizon.

    Defaults follow paper Table 8 and the pinned ``scripts/electricity.sh``.
    Small regression and smoke checks override dimensions explicitly.

    Attributes:
        history: Number of historical quarter-hours ``L``.
        horizon: Number of future quarter-hours ``H``.
        known_channels: Number of ``ForecastBatch.known_*`` channels.
        observed_channels: Number of history-only observed channels.
        width: Hidden width of the native encoder and decoder residual blocks.
        layers: Encoder and decoder block count.  ``1`` is the upstream
            single-Dense case.  ``2`` is one residual block.
        dropout: Dropout probability on each residual block output.
        decoder_output_dim: Native decoder output width ``p`` per horizon.
        temporal_width: Native feature-projection width.
        time_encoder_width: Hidden width of the feature projection block.
        final_decoder_hidden: Hidden width of the temporal final decoder.
        layer_norm: Apply Keras-style layer normalization in residual blocks.
        reversible_norm: Apply the upstream per-series affine reversible
            normalization before the encoder and invert it after the decoder.
        num_series: Number of series identifiers in the static embedding.
        categorical_sizes: Category counts for optional categorical covariates.
        categorical_embedding: Native categorical embedding width.
        series_embedding: Static time-series embedding width.  TiDE uses 16.
        layer_norm_eps: Keras ``LayerNormalization`` default epsilon.
        norm_eps: Upstream inverse-normalization epsilon.
        history_residual_lags: Most recent target steps fed to the linear skip.
            ``0`` feeds every lag, which is the upstream parity case; a small
            value drops distant pre-June spike echoes from the median path.
        pre_norm: Normalize residual-block inputs instead of outputs and use
            GELU, which keeps a common-mode encoder input from saturating.
        asof_state_channels: Steps scanned per channel when extracting the
            publication-aligned as-of state token.
        use_asof_token: Add the as-of state side-channel to the encoding.
        use_market_rule_head: Use the reserve-spread directional mixture head.
        short_history: Native-path steps; zero keeps the original full window.
            A nonzero value adds a separately patched daily context path.
        context_patch_size: Number of chronological steps per context patch.
        history_skip_asinh: Compress only the linear skip input, not the target.
        use_volume_rule_head: Blend predicted mFRR/aFRR component magnitudes
            with positive activation-volume weights before the direction gate.
        supervise_afrr: Supervise the aFRR auxiliary components.  ``False``
            restricts auxiliary supervision to the mFRR pair.
    """

    history: int = 720
    horizon: int = 40
    known_channels: int = 0
    observed_channels: int = 0
    width: int = 1024
    layers: int = 2
    dropout: float = 0.5

    decoder_output_dim: int = 8
    temporal_width: int = 4
    time_encoder_width: int = 64
    final_decoder_hidden: int = 64
    layer_norm: bool = True
    reversible_norm: bool = False
    num_series: int = 1
    categorical_sizes: tuple[int, ...] = ()
    categorical_embedding: int = 4
    series_embedding: int = 16
    layer_norm_eps: float = 1e-3
    norm_eps: float = 1e-7
    history_residual_lags: int = 0
    pre_norm: bool = False
    asof_state_channels: int = 16
    use_asof_token: bool = False
    use_market_rule_head: bool = False
    short_history: int = 0
    context_patch_size: int = 4
    history_skip_asinh: bool = False
    use_volume_rule_head: bool = False
    direct_rule_gaps: bool = False
    center_rule_anchor: bool = False
    learn_volume_weights: bool = True
    rule_gate_bias: float | None = None
    rule_aux_weight: float = 0.15
    supervise_afrr: bool = True

    def __post_init__(self):
        if self.history <= 0 or self.horizon <= 0:
            raise ValueError("history and horizon must be positive")
        if self.width <= 0 or self.layers <= 0:
            raise ValueError("width and layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.known_channels < 0 or self.observed_channels < 0:
            raise ValueError("channel counts must be non-negative")
        if self.decoder_output_dim <= 0 or self.temporal_width <= 0:
            raise ValueError("decoder_output_dim and temporal_width must be positive")
        if self.time_encoder_width <= 0 or self.final_decoder_hidden <= 0:
            raise ValueError("native hidden widths must be positive")
        if self.num_series <= 0:
            raise ValueError("num_series must be positive")
        if self.series_embedding != 16:
            raise ValueError("TiDE uses a 16-dimensional static series embedding")
        if any(size <= 0 for size in self.categorical_sizes):
            raise ValueError("categorical sizes must be positive")
        if not 0 <= self.history_residual_lags <= self.history:
            raise ValueError("history_residual_lags must be in [0, history]")
        if self.asof_state_channels <= 0:
            raise ValueError("asof_state_channels must be positive")
        if not 0 <= self.short_history <= self.history:
            raise ValueError("short_history must be in [0, history]")
        if self.context_patch_size <= 0 or (self.short_history and self.history % self.context_patch_size):
            raise ValueError("Context history must be divisible by context_patch_size")
        if self.short_history and self.history_residual_lags > self.short_history:
            raise ValueError("History skip cannot exceed short_history")
        if self.use_volume_rule_head and not self.use_market_rule_head:
            raise ValueError("Volume rule requires the market-rule head")
