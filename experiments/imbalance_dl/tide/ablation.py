"""The agreed, fixed-seed MR-TiDE architecture conditions.

``A0``-``A6`` are the first-round ladder.  ``B0``-``B5`` start from the ``A5``
candidate and change the skip, the native history path,
and the auxiliary supervision scope.  ``B0`` is exactly ``A5``.
"""

from dataclasses import replace

A_GROUPS = tuple(f"A{index}" for index in range(7))
B_GROUPS = tuple(f"B{index}" for index in range(6))
GROUPS = A_GROUPS + B_GROUPS


def _a_config(config, index):
    return replace(
        config, history=720 if index < 2 else (16 if index == 2 else 96),
        use_market_rule_head=True,
        short_history=16 if index >= 3 else 0,
        direct_rule_gaps=index >= 1,
        history_skip_asinh=index >= 4,
        use_volume_rule_head=index >= 5,
        learn_volume_weights=index >= 6,
        # Correct the same spread-unit offset and gate initialization in every group.
        center_rule_anchor=True, rule_gate_bias=-2.0,
        rule_aux_weight=0.075 if index >= 5 else 0.15,
    )


def _b_config(config, index):
    """One-factor variants around ``B0``, which equals ``A5``."""
    # A5: native 16-step path, daily 96-step context in 4-step patches, asinh
    # skip, four fixed-weight components, direct gaps, 0.075 over all components.
    base = replace(
        config, history=96, use_market_rule_head=True, short_history=16,
        direct_rule_gaps=True, history_skip_asinh=True,
        use_volume_rule_head=True, learn_volume_weights=False,
        center_rule_anchor=True, rule_gate_bias=-2.0,
        rule_aux_weight=0.075, supervise_afrr=True,
    )
    if index == 0:
        return base
    if index == 1:
        return replace(base, history_skip_asinh=False)
    # The long path keeps the full 720-step encoder input but drops the daily
    # context path and feeds only the last 16 lags to the linear skip.
    if index in (2, 3):
        long_path = replace(base, history=720, short_history=0,
                            history_residual_lags=16)
        return (long_path if index == 2
                else replace(long_path, history_skip_asinh=False))
    if index == 4:
        return replace(base, supervise_afrr=False, rule_aux_weight=0.15)
    # A zero coefficient drops the auxiliary term, so every supervised switch
    # stays irrelevant.
    return replace(base, rule_aux_weight=0.0)


def ablation_config(config, group):
    if group in B_GROUPS:
        return _b_config(config, B_GROUPS.index(group))
    return _a_config(config, A_GROUPS.index(group))
