"""Model loading and per-step canvas recording.

VERIFIED against the installed transformers/models/diffusion_gemma source
(generation_diffusion_gemma.py, generation/streamers.py):
  1. `put_draft(value, ...)` is called once per denoising step with
     `value = argmax_canvas.cpu()`, a `(batch_size, canvas_length)` LongTensor
     of token ids -- the argmax over the (temperature-scaled) denoiser
     logits, NOT decoded text. It is the FULL canvas each step (every
     position, not just newly-accepted ones). `TextDiffusionStreamer`
     (and therefore `CanvasRecorder`) only supports batch_size == 1.
  2. There is NO single mask/noise token id. `EntropyBoundSampler.
     initialize_canvas` fills un-denoised positions with i.i.d. samples from
     `torch.randint(0, vocab_size, ...)` -- uniform noise over the full
     vocabulary, redrawn every step for every not-yet-accepted position via
     `renoise_canvas`. This is a uniform-noise diffusion model, not an
     absorbing/[MASK] one.
  3. `decoder_input_ids`, if passed to `generate(...)`, is consumed verbatim
     as the starting canvas for the FIRST canvas block's denoising loop only
     (`_prepare_denoiser_inputs` pops it from `model_kwargs`; later blocks
     always start from a fresh random canvas). The denoising step loop still
     runs the FULL `max_denoising_steps` starting at `cur_step ==
     max_denoising_steps` (i.e. `t_max`), regardless of how "denoised" the
     injected canvas already is -- the temperature schedule is keyed off the
     step index, not off canvas noise level. So passing a partially-denoised
     canvas "continues" denoising in the sense that accept/renoise logic
     still runs on it (already-good tokens can still be accepted quickly if
     the model is confident), but the temperature will start high (t_max) as
     if the canvas were fully noised. To continue "sanely" from an injection
     point, `max_denoising_steps`/`t_max`/`t_min` likely need to be
     overridden to match the remaining schedule, or the loop should be
     copied into a custom function (see README checklist).
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


class CanvasRecorder(TextDiffusionStreamer):
    """Records every intermediate draft canvas during denoising.

    Usage:
        rec = CanvasRecorder(tokenizer=processor.tokenizer)
        out = model.generate(**inputs, streamer=rec, ...)
        rec.draft_history  # list[list[int]] -- token ids per denoising step
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.draft_history = []

    def put_draft(self, value, *args, **kwargs):
        # `value` is `argmax_canvas.cpu()`, a (batch_size, canvas_length) LongTensor of
        # token ids (the argmax over the denoiser logits, not decoded text). Only
        # batch_size == 1 is supported here (same constraint as the parent streamer),
        # so drop the batch dim to store a flat list[int] per step.
        ids = value[0].tolist()
        self.draft_history.append(ids)
        # Keep parent behaviour (console streaming) working:
        return super().put_draft(value, *args, **kwargs)


def build_inputs(processor, question: str, device):
    """Chat-format a GSM8K question. Keep the prompt NATURAL -- do not use an
    'Answer: __ then reasoning' template, that would manufacture the phenomenon
    we want to measure."""
    messages = [{
        "role": "user",
        "content": (
            f"{question}\n\n"
            "Think step by step, then state the final answer on the last line "
            "as: The answer is <number>."
        ),
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}
