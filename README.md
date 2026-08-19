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
- [x] `intervene_swap.py` rewritten to use `custom_denoise.run_denoising`
      — resolved (2026-08-14). It reseeds identically to the cached
      trajectory (`generate_trajectories.py` now records a per-problem
      `seed`) and replays via `run_denoising`, editing the LIVE
      `current_canvas` at the injection step rather than the cached
      argmax snapshot (splicing into the argmax snapshot and resuming from
      it — the old behavior — silently replaced real noise at not-yet-
      accepted positions with the model's own confident guess). Before
      editing, verifies the replay's `argmax_canvas` matches the cached
      trajectory at that step and raises/logs `ReplayMismatch` rather than
      intervening on a diverged run if it doesn't (relevant given the
      MoE/`torch.histc` non-determinism noted above). `matched_wrong_answer`
      now matches TOKEN length via the tokenizer, not digit-string length,
      and `splice_answer` refuses (returns `None`) rather than silently
      producing a canvas of the wrong length.
      End-to-end validated on 3 real GSM8K problems (9 live intervention
      runs: 3 injection fractions × swap/noop/random) — 0 replay mismatches.
- [x] `reasoning_converge_step` (`parse_commitment.py`) actually excludes
      the answer span and post-EOS canvas positions — resolved (2026-08-14).
      The docstring always claimed this; the code didn't do it. Left in,
      the answer span (stable early by definition) inflates the match
      fraction, and post-EOS filler (which can stay high-entropy far
      longer than real content) drags it down — both distort the
      commit-to-converge lag this metric exists to measure. Needs the
      tokenizer now (`AutoTokenizer.from_pretrained`), not just the cached
      JSON — still no GPU/model weights.
- [x] `judge.py` grades every condition, not just `swap` — resolved
      (2026-08-14), so the noop/random control rates this README's
      "Analysis endpoints" section calls for can actually be computed.
      Found and fixed a companion bug while doing this: `intervene_swap.py`
      was labeling `noop`/`random` rows with the `swap` candidate's
      `injected_answer` (shared `meta` dict across conditions) — each
      condition now gets its own accurate label.

## Analysis endpoints

Headline plot: P(reverted) / P(anchored) / P(copied) vs injection fraction.
Baselines to add before writeup: AR Gemma answer-prefill rationalization rate;
noop and random control rates from step 3.

## Code-repair feasibility pilot

`src/code_repair_pilot.py` tests a single-canvas code-generation hypothesis
before attempting internal activation patching: after an intervention makes an
otherwise-free local identifier inconsistent with its later uses, does
DiffusionGemma restore that one token, coherently propagate the rename, or
globally rewrite the program?

```bash
python src/code_repair_pilot.py --n-seeds 5 --overwrite
```

It generates ordinary short Python functions, retains only executable clean
rollouts with a repeated local identifier, and then replays each trajectory
with canvas-only, self-conditioning-only, and joint definition-site renames.
All outputs are restricted to one 256-token canvas. Results go to
`results/code_repair_pilot.jsonl`; raw trajectories go to
`data/code_repair_pilot/`.

## Live codebase map

Run this from the repo root:

```bash
python tools/codebase_viz.py
```

Open `http://127.0.0.1:8765`. It shows the experiment pipeline, the
DiffusionGemma denoising loop, and a live map of `src/*.py`. It refreshes
when a source file is saved. On a remote pod, forward the port with:

```bash
