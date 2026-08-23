"""Model loading, chat formatting, and token-span location.

VERIFIED against the installed transformers/models/diffusion_gemma source
(generation_diffusion_gemma.py, generation/streamers.py):

  1. There is NO single mask/noise token id. `EntropyBoundSampler.
     initialize_canvas` fills un-denoised positions with i.i.d. samples from
     `torch.randint(0, vocab_size, ...)` -- uniform noise over the full
     vocabulary, redrawn every step for every not-yet-accepted position via
     `renoise_canvas`. This is a uniform-noise diffusion model, not an
     absorbing/[MASK] one.
  2. `TextDiffusionStreamer.put_draft(value, ...)` is called once per
     denoising step with `value = argmax_canvas.cpu()`, the FULL canvas each
     step. `CanvasRecorder` below wraps it for the parity self-test in
     custom_denoise.py; the experiment scripts read canvases from
     `run_denoising` instead, which also exposes the internal state.
  3. Module paths for hooks live in config.py and are resolved by
     `get_module`, so a rename fails loudly rather than silently.
"""

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion, TextDiffusionStreamer

from config import MODEL_ID


def load_model(dtype=torch.bfloat16):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, processor


def get_module(model, dotted_path: str):
    """Resolve a dotted module path from config, raising a clear error."""
    module = model
    for part in dotted_path.split("."):
        if not hasattr(module, part):
            raise AttributeError(
                f"module path '{dotted_path}' broke at '{part}'. The model "
                f"layout changed; update the *_PATH values in config.py."
            )
        module = getattr(module, part)
    return module


class CanvasRecorder(TextDiffusionStreamer):
    """Records every intermediate draft canvas during `model.generate`.

    Only used by the parity self-test. `verbose=False` suppresses the parent
    streamer's live ANSI redraw, which is unreadable across a batch run.
    """

    def __init__(self, tokenizer, verbose: bool = False, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.draft_history = []
        self.verbose = verbose

    def put_draft(self, value, *args, **kwargs):
        self.draft_history.append(value[0].tolist())
        if self.verbose:
            return super().put_draft(value, *args, **kwargs)

    def put(self, value):
        if self.verbose:
            return super().put(value)

    def end(self):
        if self.verbose:
            return super().end()


def build_chat_inputs(processor, user_content: str, device):
    messages = [{"role": "user", "content": user_content}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def find_first_token_span(tokenizer, token_ids, needle: str):
    """Token span (start, end) of the FIRST occurrence of `needle`, or None.

    Builds character offsets from cumulative FULL-PREFIX decodes
    (`tokenizer.decode(token_ids[:j])` for each j), not from decoding each
    token individually and concatenating. The two differ: many tokenizers
    encode leading-space/merge information contextually, so a token's
    standalone decode is not its contribution to the joint decode. Decoding
    growing prefixes sidesteps that -- O(n^2) decode calls, but n <= 256.
    """
    prefixes = [""]
    for j in range(1, len(token_ids) + 1):
        prefixes.append(tokenizer.decode(token_ids[:j]))
    k = prefixes[-1].find(needle)
    if k < 0:
        return None
    start_char, end_char = k, k + len(needle)
    span = [
        j for j in range(len(token_ids))
        if len(prefixes[j]) < end_char and len(prefixes[j + 1]) > start_char
    ]
    return (min(span), max(span) + 1) if span else None
