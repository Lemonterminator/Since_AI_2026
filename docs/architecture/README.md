# Vartti architecture diagrams

These diagrams show the Blend B forecasting system and the four neural backbones in its ensemble. Follow a link to open the full-size SVG.

| Diagram | What it shows |
| --- | --- |
| [Blend B overview](blend_b_architecture.svg) | Publication-time input checks, the LightGBM and neural branches, forecast blending, validation metrics, and the Vartti dashboard |
| [TabularResNet](arch_1_tabular_resnet.svg) | Independent forecasts at each delivery step using Fourier features and residual blocks |
| [TiDE trajectory](arch_2_tide.svg) | Joint encoding of known covariates and a separate decoder for each step of the horizon |
| [Conv-BiGRU](arch_3_conv_bigru.svg) | Convolutions at two scales, origin state conditioning, and a bidirectional GRU |
| [Horizon Transformer-MoE](arch_4_transformer_moe.svg) | Attention over the delivery horizon and a learned mixture of four experts |

Model details are in [models.py](../../research/imbalance-forecast/dl/models.py). Metrics shown in the diagrams are validation results. The diagrams contain no market observations or trained parameters.
