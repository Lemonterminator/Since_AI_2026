"""Inference and synthetic smoke checks. This entry point never trains."""
import argparse
import json

import pandas as pd
import torch

from .shared.contracts import Forecast
from .shared.data import BatchBuilder, Feature, PRICE, RobustScaler, StandardScaler, load_gold
from .shared.export import apply_interval_scales, forecast_frame, write_submission
from .shared.inputs import prepare_inputs


def build_model(name, config):
    if name == "tide":
        from .tide import TiDEConfig, TiDEForecaster
        return TiDEForecaster(TiDEConfig(**config))
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def smoke(name):
    from .tests.fixtures import synthetic_batch
    names = (name,)
    for track in names:
        torch.manual_seed(20261107)
        batch, targets = synthetic_batch(batch_size=2, history=96, horizon=40)
        config = dict(history=96, horizon=40, known_channels=3, observed_channels=2,
                      width=16, layers=1, dropout=0.)
        model = build_model(track, config).eval()
        prediction = model(batch)
        losses = model.objective(batch, targets)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise RuntimeError(f"Non-finite {track} loss")
        frame = forecast_frame(prediction, batch)
        print(json.dumps({"model": track, "quantiles": list(prediction.quantiles.shape),
                          "rows": len(frame), "loss": float(losses["total"]),
                          "parameter_updates": 0, "device": "cpu"}))


@torch.no_grad()
def predict(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(args.model, checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(args.device).eval()
    features = tuple(Feature(**item) for item in checkpoint["features"])
    multi_market = checkpoint["scaler"].get("kind") == "multi_market"
    if multi_market:
        from .shared.markets import MultiMarketBuilder
        if args.model != "tide":
            raise ValueError("Multi-market checkpoints require TiDE")
        builder = MultiMarketBuilder.from_state_dict(load_gold(args.gold), checkpoint["scaler"])
        if tuple(features) != builder.features:
            raise ValueError("Checkpoint feature specifications disagree")
        if "FI" not in builder.markets:
            raise ValueError("Official prediction requires an FI target")
    else:
        scaler_type = StandardScaler if args.model == "tide" else RobustScaler
        scaler = scaler_type(**checkpoint["scaler"])
        events = pd.read_parquet(args.events) if args.events else None
        gold = prepare_inputs(load_gold(args.gold), checkpoint.get("price_series", PRICE))
        builder = BatchBuilder(gold, features, scaler,
                               history=model.config.history, horizon=model.config.horizon, events=events)
    # Calibrated interval factors live in the checkpoint metadata; absent is a no-op.
    scales = (checkpoint.get("metadata") or {}).get("interval_scales")
    origins = pd.read_csv(args.origins).forecast_origin_utc.tolist()
    frames = []
    for start in range(0, len(origins), args.batch_size):
        selected = origins[start:start + args.batch_size]
        samples = [("FI", origin) for origin in selected] if multi_market else selected
        batch = builder.build(samples).to(args.device)
        output = model(batch)
        if scales:
            output = Forecast(apply_interval_scales(output.quantiles, scales), output.direction,
                              output.spikes, output.samples)
        frames.append(forecast_frame(output, batch))
    if not frames:
        raise ValueError("The origins file is empty")
    counts = write_submission(pd.concat(frames, ignore_index=True), args.output, args.origins)
    print(json.dumps(counts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("smoke", help="Synthetic forward/loss checks without parameter updates")
    check.add_argument("--model", choices=("tide",), default="tide")
    inference = subcommands.add_parser("predict", help="Load an existing checkpoint and export validated predictions")
    inference.add_argument("--model", choices=("tide",), default="tide")
    for name in ("checkpoint", "origins", "output"):
        inference.add_argument("--" + name, required=True)
    inference.add_argument("--gold", help="Gold CSV or extended Parquet; defaults to the existing external layer")
    inference.add_argument("--events", help="Existing nuclear_asof.parquet for configured event features")
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "smoke":
        smoke(args.model)
    else:
        if min(args.batch_size, args.samples, args.chunk_size) <= 0:
            parser.error("Batch size and sample counts must be positive")
        predict(args)


if __name__ == "__main__":
    main()
