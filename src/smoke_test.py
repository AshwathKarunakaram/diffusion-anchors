"""Hour-1 smoke test. Run this FIRST, inside tmux.

Checks, in order:
  1. model loads on the A100 (bf16, ~50GB)
  2. one generation completes
  3. CanvasRecorder captures per-step drafts
  4. prints the installed source paths you must read before trusting the
     scaffold's assumptions (put_draft signature, sampler, decoder_input_ids)
"""

import torch

from model_utils import load_model, CanvasRecorder, build_inputs


def main():
    model, processor = load_model()
    print(f"loaded; GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    inputs = build_inputs(processor,
                          "Natalia sold clips to 48 friends and then half as "
                          "many again. How many clips did she sell in total?",
                          model.device)
    rec = CanvasRecorder(tokenizer=processor.tokenizer)
    out = model.generate(**inputs, max_new_tokens=256, streamer=rec)

    gen = processor.tokenizer.decode(
        out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n--- final output ---\n", gen)
    print(f"\ncaptured {len(rec.draft_history)} intermediate canvases")
    if rec.draft_history:
        # Each entry in `draft_history` is now a flat list[int] of token ids
        # (argmax over that step's denoiser logits), one per canvas position --
        # see CanvasRecorder.put_draft. Un-accepted positions are still i.i.d.
        # uniform-random token ids (no fixed mask id), so decoding the midpoint
        # canvas will show a mix of committed text and noise tokens; that's
        # expected, not a bug.
        mid = rec.draft_history[len(rec.draft_history) // 2]
        print("--- canvas at midpoint ---\n", processor.tokenizer.decode(mid))

    # Source files to read next (this is where your assumptions get checked):
    import transformers.models.diffusion_gemma as dg
    print("\nREAD THESE FILES NEXT:")
    print(" ", dg.__file__)
    import inspect
    from transformers import TextDiffusionStreamer
    print("\nput_draft signature:",
          inspect.signature(TextDiffusionStreamer.put_draft))


if __name__ == "__main__":
    main()
