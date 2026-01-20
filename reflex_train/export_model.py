import argparse
import json
import os
import torch
import torch.nn as nn
from reflex_train.models.reflex_net import ReflexNet
from reflex_train.data.keys import NUM_KEYS, TRACKED_KEYS


class ReflexNetExport(nn.Module):
    """Wrapper for ONNX export that returns only the outputs we need."""

    def __init__(self, model: ReflexNet):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        keys_logits, mouse_pos, _, value = self.model(pixels)
        return keys_logits, mouse_pos, value


def build_model(checkpoint_path, use_random):
    model = ReflexNet(
        context_frames=3,
        num_keys=NUM_KEYS,
        inv_dynamics=False,
    )
    if not use_random:
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path or "<missing>")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    return model


def write_manifest(output_path, height, width):
    manifest = {
        "inputs": {
            "pixels": {
                "shape": ["batch", 9, "height", "width"],
                "order": "RGB",
                "stack_order": ["t-2", "t-1", "t"],
                "scale": "0..1",
                "height": height,
                "width": width,
            },
        },
        "outputs": {
            "keys_logits": {
                "shape": ["batch", NUM_KEYS],
                "keys": TRACKED_KEYS,
            },
            "mouse_pos": {
                "shape": ["batch", 2],
                "range": "0..1",
            },
            "value": {
                "shape": ["batch"],
                "description": "Expected return estimate (higher = better state)",
            },
        },
        "notes": [
            "pixels are stacked as [t-2, t-1, t], each RGB",
            "value is the estimated return from the current state",
        ],
    }
    manifest_path = os.path.splitext(output_path)[0] + "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


def export_onnx(checkpoint_path, output_path, height, width, use_random):
    model = build_model(checkpoint_path, use_random)
    export_model = ReflexNetExport(model)
    export_model.eval()

    pixels = torch.zeros(1, 9, height, width, dtype=torch.float32)
    dynamic_axes = {
        "pixels": {0: "batch", 2: "height", 3: "width"},
        "keys_logits": {0: "batch"},
        "mouse_pos": {0: "batch"},
        "value": {0: "batch"},
    }

    torch.onnx.export(
        export_model,
        (pixels,),
        output_path,
        input_names=["pixels"],
        output_names=["keys_logits", "mouse_pos", "value"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )
    print(f"Wrote {output_path}")
    write_manifest(output_path, height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    if not args.random and not args.checkpoint:
        raise SystemExit("Provide --checkpoint or use --random for a dummy model.")

    export_onnx(args.checkpoint, args.output, args.height, args.width, args.random)


if __name__ == "__main__":
    main()
