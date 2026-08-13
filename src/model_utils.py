"""Model loading and per-step canvas recording.

VERIFY-ON-POD list (API details taken from HF docs for transformers'
diffusion_gemma; confirm against your installed version before trusting):
  1. Exact signature of TextDiffusionStreamer.put_draft():
       python -c "import inspect; from transformers import TextDiffusionStreamer; print(inspect.getsource(TextDiffusionStreamer))"
  2. What token id fills un-denoised canvas positions (mask/noise token).
  3. Whether put_draft receives the FULL canvas each step or only changed tokens.
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
        try:
            ids = value.detach().to("cpu").tolist()
        except AttributeError:
            ids = list(value)
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
