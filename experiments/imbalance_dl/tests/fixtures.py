"""Small synthetic contracts. No training data or parameter updates."""
import torch

from ..shared.contracts import ForecastBatch, Targets


def synthetic_batch(batch_size=2, history=8, horizon=4, observed=2, known=3):
    generator = torch.Generator().manual_seed(20261107)
    def random(*shape):
        return torch.randn(*shape, generator=generator)
    target, observations = random(batch_size, history, 1), random(batch_size, history, observed)
    past, future = random(batch_size, history, known), random(batch_size, horizon, known)
    masks = [torch.ones_like(x, dtype=torch.bool) for x in (target, observations, past, future)]
    ages = [torch.zeros_like(x) for x in (target, observations, past, future)]
    origins = tuple(f"2026-09-01T{i:02d}:00:00Z" for i in range(batch_size))
    batch = ForecastBatch(target, observations, past, future, *masks, *ages,
                          torch.full((batch_size, horizon), 25.0), torch.zeros(batch_size, 1),
                          torch.full((batch_size, 1), 100.0), origins, origins,
                          torch.ones(observed, dtype=torch.bool), torch.ones(known, dtype=torch.bool))
    spread = random(batch_size, horizon)
    raw_spread = spread * batch.spread_scale + batch.spread_center
    labels = Targets(spread, torch.zeros(batch_size, horizon, dtype=torch.long),
                     torch.ones(batch_size, horizon, dtype=torch.bool), torch.ones(batch_size, horizon, dtype=torch.bool),
                     torch.stack((raw_spread > 50, raw_spread < -50), -1))
    return batch, labels
