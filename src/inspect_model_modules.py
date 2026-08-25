"""Print the DiffusionGemma module tree the lens capture script must hook.

Loads the model (weights already cached on the runtime), prints:
  * config fields that size the lens (layers, hidden dim, vocab, canvas);
  * every named module to a limited depth, with class names;
  * the ModuleLists that look like transformer-layer stacks;
  * output-embedding / final-norm candidates for the logit-lens projection.

No generation is run. Paste the output back so the capture hooks target
real module paths instead of guesses.

Run:  python src/inspect_model_modules.py
"""

import torch.nn as nn

from model_utils import load_model

MAX_DEPTH = 4


def main():
    model, _ = load_model()

    print("=== config ===")
    config = model.config
    for key in sorted(vars(config).keys() | set(dir(config))):
        if key.startswith("_"):
            continue
        if any(tag in key.lower() for tag in (
            "layer", "hidden", "vocab", "canvas", "head", "denois", "embed",
            "conditioning", "expert", "moe",
        )):
            value = getattr(config, key, None)
            if isinstance(value, (int, float, str, bool, list, tuple)) or value is None:
                print(f"{key} = {value}")

    print("\n=== module tree (depth <= {0}) ===".format(MAX_DEPTH))
    for name, module in model.named_modules():
        depth = name.count(".")
        if name and depth <= MAX_DEPTH:
            size = ""
            if isinstance(module, nn.ModuleList):
                size = f" [len={len(module)}]"
            print(f"{'  ' * depth}{name}: {type(module).__name__}{size}")

    print("\n=== transformer-stack candidates (ModuleLists of repeated blocks) ===")
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            child_types = {type(child).__name__ for child in module}
            print(f"{name}: len={len(module)} child_types={sorted(child_types)}")

    print("\n=== logit-lens projection candidates ===")
    out_embed = model.get_output_embeddings()
    print(f"get_output_embeddings(): {type(out_embed).__name__} "
          f"weight={tuple(out_embed.weight.shape) if out_embed is not None else None}")
    for name, module in model.named_modules():
        lowered = name.lower()
        if any(tag in lowered for tag in ("norm", "lm_head", "final")) and name.count(".") <= 3:
            print(f"{name}: {type(module).__name__}")


if __name__ == "__main__":
    main()
