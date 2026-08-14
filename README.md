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
- [ ] `decoder_input_ids` continues from a partially-denoised canvas sanely
      (read the sampler: `initialize_canvas` / `renoise_canvas` / `accept_canvas`)
      — **partially wrong as-is**: `generate(decoder_input_ids=...)` is
      consumed verbatim as the starting canvas for the denoising loop
      (`_prepare_denoiser_inputs` pops it from `model_kwargs`), and only for
      the *first* canvas block — later blocks always start from a fresh
      random canvas, so `decoder_input_ids` can't be used to inject into a
      later block directly. Accept/renoise logic (entropy-based) runs
      correctly on any canvas contents, but `cur_step` still starts at
      `max_denoising_steps` regardless of how denoised the injected canvas
      already is, so the temperature schedule starts at `t_max` (as if fully
      noised) instead of at the value matching the injection point.
- [ ] temperature schedule on continuation adjusted to injection-step value
      — not yet done; needs either overriding `max_denoising_steps`/`t_max`/
      `t_min` per-call to match the remaining schedule, or a custom loop (see
      next item).
- [ ] if `decoder_input_ids` semantics are wrong → copy the generation loop
      into a custom function (also unlocks self-conditioning patching later)
      — **required**: given the above, `intervene_swap.py` should copy
      `DiffusionGemmaGenerationMixin.generate`'s inner denoising loop
      (`_denoising_step` et al.) rather than rely on `decoder_input_ids`,
      so `cur_step`/temperature can be pinned to the injection point and
      `self_conditioning_logits` can be patched directly.

## Analysis endpoints

Headline plot: P(reverted) / P(anchored) / P(copied) vs injection fraction.
Baselines to add before writeup: AR Gemma answer-prefill rationalization rate;
noop and random control rates from step 3.
