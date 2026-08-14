# Project context — read this before doing anything else

Paste/commit this file as `CONTEXT.md` in the repo root. Point Cursor/Codex at
it explicitly ("read CONTEXT.md first") so you don't have to re-explain the
project each session.

## The research question

MATS application project for Neel Nanda's stream (~20h budget). Hypothesis:

> DiffusionGemma often commits to an answer early in denoising, before
> reasoning is finished. Because DiffusionGemma (unlike LLaDA/Dream) can
> revise any token at any step — non-selected positions are fully renoised
> each step, nothing is permanently frozen — an early commitment is always
> *revisable*. We ask: once committed, does the model keep the answer and
> construct supporting reasoning around it (causal anchoring /
> rationalization), or does it revert to a different answer if perturbed?
> And how does this change as a function of when in the trajectory we
> intervene?

Falsification is cheap: soft-swap the committed answer to a matched wrong
value mid-trajectory, let denoising continue, see what comes out. Read
`README.md` for the full experiment design and run order.

## Nearby literature (already checked — do not re-litigate novelty)

- **arXiv 2608.05687 "Answer First, Reason Later"** — closest paper. Studies
  LLaDA-8B/Dream-7B, shows answer commits before reasoning (median step
  ~0.15-0.24 vs ~0.5), reasoning is "answer-conditioned" post-commitment. But:
  (a) those models freeze committed tokens by construction — revision is
  impossible, so "conditioning" is partly mechanical; (b) **no answer
  injection/patching/counterfactual-forcing experiment exists in that paper**
  (confirmed by full read incl. all 5 appendices) — it's observational +
  one decoding-order intervention (frontier-gated commitment window), nothing
  else. Our swap experiment is the missing causal arrow. DiffusionGemma is
  not studied there at all.
- **"Inception in DiffusionGemma" (LessWrong jailbreak post)** — already did
  soft/hard token pinning on DiffusionGemma specifically (StrongREJECT
  prompts, not math/reasoning). Found soft-pinned tokens are often *retained*
  even though revisable ("tends to stay on that path"). This raises the
  prior that our phenomenon is real and de-risks the intervention mechanics,
  but has no answer/reasoning structure or time-resolved sweep.
- **arXiv 2508.19982** "DLMs Know the Answer Before Decoding" — established
  the *behavioral* early-convergence fact (why "does the answer appear
  early" alone is not novel).
- **How Transparent is DiffusionGemma? (arXiv 2606.20560, Neel's paper)** —
  found ~90% of intermediate self-conditioning-derived tokens match final
  canvas tokens (self-conditioning ≈ token summary by default); documented
  retroactive self-correction as a real phenomenon. Section 7 open problems
  this project sits at the intersection of: activation patching across
  diffusion steps, intermediate-vs-final tokens, post-hoc
  reasoning/unfaithfulness, Thought-Anchor-style resampling.
- Mandatory in the writeup: cite 2608.05687 and the Inception post yourself,
  in the first paragraph, and state the revisability distinction explicitly.

## Model facts (verified by reading installed transformers source, not docs)

- `google/diffusiongemma-26B-A4B-it`, MoE, 25.2B total / 3.8B active params,
  Apache 2.0, HF `transformers` support via `DiffusionGemmaForBlockDiffusion`.
- **Uniform-noise diffusion, not absorbing/mask-token.** Undecided canvas
  positions hold i.i.d. `torch.randint(0, vocab_size)` — redrawn every step
  for unaccepted positions (`EntropyBoundSampler.renoise_canvas`). There is
  no fixed [MASK] id. What you see mid-trajectory is the model's per-step
  argmax guess at every position, including ones it hasn't "decided" yet —
  so **commitment must be defined as stability across steps, not first
  appearance.** (Already implemented this way in `parse_commitment.py`.)
- Canvas length 256, up to 48 denoising steps, temperature schedule
  t_max=0.8 → t_min=0.4, entropy-bound acceptance + stability-based early
  stop.
- `put_draft(value, ...)` is called every step with
  `value = argmax(processed_logits)`, shape `(batch=1, canvas_length)` —
  full canvas each step, not a diff. **Bug found & fixed**: `CanvasRecorder`
  wasn't stripping the batch dim (was storing `list[list[list[int]]]`,
  should be `list[list[int]]` — one flat token-id list per step). Fixed in
  `src/model_utils.py` (`value[0].tolist()`).
- **`decoder_input_ids` is NOT sound for mid-trajectory injection.** Confirmed
  by reading `generation_diffusion_gemma.py`: a provided canvas is consumed
  verbatim as the *starting* canvas, but `cur_step` always resets to
  `max_denoising_steps` — i.e. the temperature schedule restarts at t_max
  regardless of how denoised the injected content already is. Passing an
  edited step-30-of-48 canvas via `decoder_input_ids` silently reruns it as
  if it were step 0. **Do not use this path for the intervention experiment.**
- **Resolved (2026-08-14):** `src/custom_denoise.py` exists and is parity-
  validated. Rather than retyping the library's per-step math, it calls
  `DiffusionGemmaGenerationMixin._denoising_step` directly, once per step —
  so `intervention_fn=None` structurally IS `generate()`'s math, not a
  reimplementation that can drift. Exposes a hook (`intervention_fn`) called
  between steps that can replace the canvas and/or the self-conditioning
  logits before the next step, without resetting `cur_step` — this is what
  makes mid-trajectory injection sound (temperature schedule just continues).
  Single canvas block only, batch_size==1 only (matches this project's needs
  exactly; raises `NotImplementedError` outside that scope on purpose).
  Parity check (single GSM8K problem, seed 0, `disable_compile=True`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms warn_only):
  all 9 denoising steps matched HF `generate()` exactly, decoded text
  identical through the first EOS token; only the post-EOS pad filler
  differed (expected — `_finalize_canvas`'s pad-replacement is multi-block
  bookkeeping this file intentionally doesn't copy). Also confirms self-
  conditioning-vector patching (Section 7 stretch goal, [[README]]) is
  reachable through the same hook if there's time left in the budget.
  **Caveat**: this is a 26B MoE model — `grouped_mm_experts_forward` routes
  through `torch.histc`, which has no deterministic CUDA kernel, so two
  same-seed runs aren't guaranteed bit-identical. Didn't cause a divergence
  here, but it's the first thing to suspect if a future parity check does
  diverge with no other explanation.
  **Still open:** `intervene_swap.py` still calls the old, unsound
  `decoder_input_ids` path — needs rewriting to call
  `custom_denoise.run_denoising(..., intervention_fn=...)`.

## Repo layout / run order

See `README.md` for the full table. Summary:

0. `src/smoke_test.py` — sanity check, already passing (loads model,
   generates one correct GSM8K answer, captures per-step canvases).
1. `src/generate_trajectories.py` — GPU, caches per-step canvases for N
   short-solution GSM8K problems to `data/trajectories/*.json`.
   **Known issue from smoke test**: the trivially easy example problem
   converged in ~10 steps with almost no gap between mid-trajectory and
   final state — no intervention window. Problem selection needs to bias
   toward problems that actually use most of the ~48 steps (harder GSM8K,
   or filter post-hoc on trajectories with a large `commit_lead`), not just
   "short solution text" as `config.py` currently does.
2. `src/parse_commitment.py` — CPU only, computes answer-commit-step and
   reasoning-converge-step per problem, writes `results/commitment.csv`.
   **This is the premise check** — if answer rarely commits meaningfully
   before reasoning converges, the whole project pivots (see original
   research-planning conversation / Problem #2 as backup, deprioritized
   due to needing a non-differentiable RL-ish fine-tuning loop on a 26B MoE
   model in a 20h budget — avoid unless #1 truly dies).
3. `src/intervene_swap.py` — GPU, the core experiment. **Currently written
   against `decoder_input_ids`, which is confirmed unsound — needs
   refactoring to use `custom_denoise.py` once that exists and is
   validated.** Conditions per problem/injection-step: swap (treatment,
   answer → matched wrong value), noop (re-inject same answer, disruption
   control), random (perturb a random already-stable non-answer token,
   recovery control).
4. `src/judge.py` — CPU, needs `ANTHROPIC_API_KEY` (see below). Deterministic
   last-equation check + Claude-Haiku judge on whether post-injection
   reasoning supports the injected answer. Codes outcomes: reverted /
   anchored / copied / derailed. Also needs, before final writeup: an
   autoregressive-Gemma answer-prefill baseline (same swap-and-continue
   idea but on plain AR Gemma) to isolate what's diffusion-specific from
   generic LM rationalization (well-known in AR models, e.g. Turpin et al.,
   Lanham et al. — the diffusion-specific claim is the whole point, so this
   baseline is not optional for the writeup).

## Environment (RunPod A100 80GB PCIe, Secure Cloud)

Fixed already, but if the pod restarts you may need to redo some of this
(container-disk installs don't survive stop/start; `/workspace` does):

```bash
export HF_HOME=/workspace/hf   # already in ~/.bashrc — weights persist on the volume
apt-get install -y tmux        # not preinstalled on this image
pip uninstall -y torchaudio    # incompatible with upgraded torch, not needed, breaks import
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# ^ pod driver is CUDA 12.4; transformers needs torch>=2.5; plain `pip install torch` grabs
#   a cu13 build that fails with "NVIDIA driver is too old". This exact pin is what worked.
pip install -U transformers accelerate datasets anthropic matplotlib
```

tmux: `tmux new -s main` to start, `Ctrl+b c` new window, `Ctrl+b <number>`
switch, `Ctrl+b d` detach, `tmux attach` to resume. Mouse scroll enabled via
`~/.tmux.conf` (`set -g mouse on`). **Always work inside tmux** — anything
over a minute (installs, downloads, GPU runs) goes in its own window so a
dropped SSH connection doesn't kill it.

**Discipline**: Stop the pod (not Terminate) whenever not actively using the
GPU — steps 2 and 4 are CPU-only, do those with the pod stopped if trying to
save money, or just leave it running if mid-session. Never Terminate until
the project is fully done and pushed.

## Credentials

- Claude Code on the pod: logged in via `claude` + plan login (browser OAuth),
  not an API key — fine for a short-lived disposable pod, revocable via
  Claude account session settings if ever needed.
- `judge.py` needs `export ANTHROPIC_API_KEY=...` from an API console key
  (separate from plan login) — use YC/student credits if claimed. Never
  commit this key; `.gitignore` already excludes `.env` but the key should
  live only in an `export` line in the shell / `~/.bashrc`, never in a
  tracked file.
- HF token (Read-only) used once for `huggingface-cli login` to pull gated
  weights if needed.

## What "done" looks like for the core deliverable

A plot of P(reverted) / P(anchored) / P(copied) vs. injection fraction
(0.25/0.5/0.75 between commit step and reasoning-converge step), with noop
and random control rates overlaid, plus the AR-Gemma baseline rate as a
horizontal reference line. Headline claim is whichever of these is true:
sharp transition from revertible to load-bearing at some point in the
trajectory (best case), flat-high (anchoring is just conditioning, weaker
but still real finding), or flat-low (early answers are epiphenomenal,
strong negative result). All three are writeup-worthy; report whichever one
you actually get, including if it's messy/noisy — do not narrativize noise
into a clean transition that isn't really there.