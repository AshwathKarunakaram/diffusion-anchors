"""Central config for the answer lock-in project."""

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# One canvas only. Every experiment compares (layer x denoising step) within a
# single 256-token canvas; multi-canvas generation degenerates toward
# autoregressive behaviour and is out of scope. Scripts assert this value.
CANVAS_LENGTH = 256

# Model default; kept explicit so it is logged with every run.
MAX_DENOISING_STEPS = 48

# Decoder module paths, verified against the loaded model. Every hook in the
# repo resolves through these, so a transformers upgrade that renames a module
# fails loudly in one place instead of silently capturing the wrong tensor.
DECODER_LAYERS_PATH = "model.decoder.layers"
DECODER_NORM_PATH = "model.decoder.norm"
SELF_CONDITIONING_PATH = "model.decoder.self_conditioning"
