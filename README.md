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

- [ ] `TextDiffusionStreamer.put_draft` receives the full canvas each step
- [ ] noise/mask token id for un-denoised positions identified
- [ ] `decoder_input_ids` continues from a partially-denoised canvas sanely
      (read the sampler: `initialize_canvas` / `renoise_canvas` / `accept_canvas`)
- [ ] temperature schedule on continuation adjusted to injection-step value
- [ ] if `decoder_input_ids` semantics are wrong → copy the generation loop
      into a custom function (also unlocks self-conditioning patching later)

## Analysis endpoints

Headline plot: P(reverted) / P(anchored) / P(copied) vs injection fraction.
Baselines to add before writeup: AR Gemma answer-prefill rationalization rate;
noop and random control rates from step 3.
