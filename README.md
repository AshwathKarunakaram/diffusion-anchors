# diffusion-anchors

Do early answer commitments act as causal thought anchors in DiffusionGemma?

DiffusionGemma renoises unselected tokens every step, so an early-converged
answer is *revisable* — unlike LLaDA/Dream where committed tokens are frozen
(arXiv 2608.05687 showed answer-first ordering there, but never manipulated
the answer). We soft-swap the committed answer mid-denoising and measure
whether the model reverts it or builds reasoning around it, as a function of
injection time.

## Pod setup (once)

```bash
# on the pod (RunPod A100 80GB, PyTorch template), inside tmux:
tmux new -s main
git clone <your-repo-url> && cd diffusion-anchors
pip install -U torch transformers accelerate datasets anthropic matplotlib
huggingface-cli login        # HF token with gated-model access if required
export ANTHROPIC_API_KEY=... # YC credits
```

Cursor: Remote-SSH to the pod, open this folder. Run Claude Code in a tmux
window (`tmux new-window`), long GPU jobs in another.

## Run order

| # | command | GPU? | what it does / kill criteria |
|---|---------|------|------------------------------|
| 0 | `python src/smoke_test.py` | yes | model loads, recorder works; prints source files to READ before step 2 |
| 1 | `python src/generate_trajectories.py` | yes | caches per-step canvases for 50 short GSM8K problems |
| 2 | `python src/parse_commitment.py` | no | **premise check**: if answer rarely commits before reasoning converges, PIVOT |
| 3 | `python src/intervene_swap.py` | yes | soft answer-swap + noop/random controls |
| 4 | `python src/judge.py` | no | codes outcomes: reverted / anchored / copied / derailed |

## Verified-assumptions checklist (do not skip)

- [x] `TextDiffusionStreamer.put_draft` receives the full canvas each step —
      confirmed in `generation_diffusion_gemma.py`: `put_draft(value=argmax_canvas.cpu())`
      is called every denoising step with `argmax_canvas`, a full
      `(batch_size, canvas_length)` LongTensor of **token ids** (argmax over
      that step's temperature-scaled logits), not decoded text, and not a
      diff/delta — every position is included every step.
- [x] noise/mask token id for un-denoised positions identified — there isn't
      a fixed one. `EntropyBoundSampler.initialize_canvas` draws i.i.d.
      `torch.randint(0, vocab_size, ...)`, i.e. uniform noise over the full
      vocab, and `renoise_canvas` redraws fresh uniform-random ids for every
      not-yet-accepted position each step. This is a uniform-noise diffusion
      model, not an absorbing/[MASK]-token one — there's no single sentinel
      id to filter on when reading a canvas.
- [x] `decoder_input_ids` continues from a partially-denoised canvas sanely
      — confirmed unsound, same reason as before (verbatim starting canvas,
      first block only, `cur_step` always resets to `max_denoising_steps`).
      **Resolved by not using this path at all** — see next item.
- [x] temperature schedule on continuation adjusted to injection-step value
      — resolved: `src/custom_denoise.py` re-drives the real per-step
      function (`_denoising_step`) one step at a time instead of restarting
      `generate()`, so `cur_step` (and therefore the temperature) just
      continues naturally across an injection — nothing to reset or override.
- [x] if `decoder_input_ids` semantics are wrong → copy the generation loop
      into a custom function (also unlocks self-conditioning patching later)
      — **done**: `src/custom_denoise.py`. Design note: it does NOT retype
      the step math by hand — it calls `DiffusionGemmaGenerationMixin.
      _denoising_step` directly, once per step, so `intervention_fn=None`
      structurally IS `generate()`'s math, not a lookalike. Exposes a hook
      to edit the canvas (o^t) and/or self-conditioning logits (S^t) between
      any two steps.
      **Parity validated** (2026-08-14, single GSM8K problem, seed 0,
      `disable_compile=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
      `torch.use_deterministic_algorithms(True, warn_only=True)`): all 9
      denoising steps' argmax canvases matched HF `generate()` exactly, full
      decoded text identical through the first EOS token. Only difference
      was trailing post-EOS filler, which is HF's `_finalize_canvas`
      pad-replacement — multi-block bookkeeping `custom_denoise.py`
      intentionally doesn't implement (single-block only, by design).
      **Caveat for future runs**: this is a 26B MoE model; its expert
      routing (`transformers/integrations/moe.py:grouped_mm_experts_forward`)
      uses `torch.histc`, which has no deterministic CUDA kernel. Two runs
      with the same seed are not guaranteed bit-identical. If a future
      parity check shows a divergence with no other explanation, suspect
      this before suspecting the loop logic.
      **Still open**: `intervene_swap.py` still calls the old
      `decoder_input_ids` path — needs rewriting to call
      `custom_denoise.run_denoising(..., intervention_fn=...)` instead.

## Analysis endpoints

Headline plot: P(reverted) / P(anchored) / P(copied) vs injection fraction.
Baselines to add before writeup: AR Gemma answer-prefill rationalization rate;
noop and random control rates from step 3.
