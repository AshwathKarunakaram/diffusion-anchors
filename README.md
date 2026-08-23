# Answer lock-in in DiffusionGemma

When DiffusionGemma answers a question answer-first, some random seeds write a
wrong answer and never revise it, even though the reasoning printed beside it
is correct and denoising steps remain. This repo characterises that failure,
tests whether the correct answer is present internally, and asks whether the
outcome can be changed causally.

DiffusionGemma fills a 256-token canvas over repeated denoising steps rather
than writing left to right, so every token stays revisable until the end. It
also feeds its previous step's output distribution back into itself each step
through a **self-conditioning** channel — a recurrent state separate from the
visible canvas. That channel is the main object of study here; it has no
autoregressive analogue.

Everything is single-canvas, inference-only, and runs on one A100 80GB. No
finetuning.

## Findings so far

| Claim | Evidence |
|---|---|
| Lock-in replicates across prompt families | 250 trajectories, 19 families, 4 with both fates on the same prompt |
| Correction is one-way | 0/250 runs abandoned a correct answer; 100+ fixed a wrong one |
| Wrong answers are attractors, not noise | 678,320 in 7/10 seeds; 120 (= C(10,3), the include-zero error) in 6/6 wrong seeds where gold is 84 |
| Logit lens is blind below ~layer 26 | layer 29 agreement 1.000, layer 15 ≤ 0.008; mid-layers decode to cross-lingual number words |
| Internal readability predicts fate exactly | gold readable in 24/24 correcting runs, 0/17 locked runs (readable band only) |
| Self-conditioning causally carries the fate | donor transplant rescues 119/132 (90%); locked-run donor 11/80 (14%); shuffled 27/132 (20%) |
| A single pooled direction does not explain it | rank-1 steering 11/32 vs random 7/32, p = 0.20 — negative result |

Two caveats that belong with every one of these: "internally absent" means
absent from the layers a fixed unembedding can read (~26-29), and everything
is one model with short integer-answer prompts and a single canvas.

## Pipeline

Run in order. Steps marked GPU need the A100; the rest are CPU and re-run
freely. Outputs append to `results/`, caches go to `data/`, both gitignored.

| # | Command | GPU | Produces |
|---|---|---|---|
| 0 | `python src/custom_denoise.py` | yes | parity check: the instrumented loop matches stock `generate()` |
| 1 | `python src/generate_lockin_sweep.py --n-seeds 10 --overwrite` | yes | `results/lockin_sweep.jsonl`, per-step canvases |
| 2 | `python src/generate_lockin_sweep.py --n-seeds 30 --start-seed 10 --prompt two_aces_hand --prompt increasing_three_digits` | yes | deepens the two mixed families to 40 seeds |
| 3 | `python src/relabel_lockin_sweep.py` | no | rebuilds every label from cached drafts |
| 4 | `python src/capture_lens.py --smoke` | yes | hook alignment check — layer 29 must read 1.0 |
| 5 | `python src/capture_lens.py --prompt two_aces_hand --prompt increasing_three_digits --max-per-label 12` | yes | `data/lens_capture/` |
| 6 | `python src/analyze_lens.py` | no | layer × step heatmaps, gold-readability curves, events table |
| 7 | `python src/patch_lockin.py --prompt two_aces_hand --steps 1,2 --n-donors 3 --locked-donor` | yes | the causal result plus both controls |
| 8 | `python src/patch_lockin.py --prompt two_aces_hand --steps 4,6,8,10 --regions all,not_answer --skip-shuffle` | yes | point-of-no-return sweep |
| 9 | `python src/analyze_patch.py` | no | PNR curve, condition bars, exact tests |
| 10 | `python src/steer_lockin.py --smoke` | yes | self-conditioning hook alignment |
| 11 | `python src/steer_lockin.py --prompt two_aces_hand --mode perpos` | yes | steering: rescue / random / induce |
| 12 | `python src/doom_detector.py` | yes then no | probe AUC, within-prompt and cross-prompt |
| 13 | `python src/capture_routing.py --smoke` then without `--smoke` | yes | optional: expert-routing divergence |

Step 13 is exploratory and may return a null; nothing else depends on it.

## Reading the numbers

Every intervention must be read against the family's **natural correct rate**
(printed by `analyze_patch.py`), not against zero. A "the patch just restarts
the computation" explanation predicts exactly that rate, so an arm sitting at
it carries no information. This matters concretely: on `two_aces_hand` (base
72.5%) the shuffled-state control discriminates well, but on
`increasing_three_digits` (base 85%) shuffling scores 86% and is useless —
there, only the locked-donor arm separates the hypotheses.

Trajectory labels (`possible_corrector`, `possible_locked_wrong`,
`possible_lost_correct`, `always_or_early_correct`, `wrong_unstable_or_other`)
are triage from decoded text and keep "possible" deliberately: they cannot
tell a change of mind from canvas flicker. All headline statistics use exact
final answers instead.

## Layout

```
src/
  config.py                 model id, canvas length, hook module paths
  model_utils.py            loading, chat formatting, token-span search
  custom_denoise.py         instrumented denoising loop (+ parity self-test)
  lockin_prompts.py         19 answer-first prompt families with gold answers
  lockin_answers.py         answer extraction and trajectory labels
  generate_lockin_sweep.py  behavioural sweep
  relabel_lockin_sweep.py   offline relabel from cached drafts
  capture_lens.py           logit lens across (layer x denoising step)
  analyze_lens.py           lens figures and gold-visibility events
  patch_lockin.py           self-conditioning transplant + controls
  analyze_patch.py          point-of-no-return curve and exact tests
  steer_lockin.py           difference-of-means steering (pooled / per-position)
  doom_detector.py          linear probe predicting fate at an early step
  capture_routing.py        exploratory MoE routing divergence
notebooks/run_all.ipynb     Colab runner with Drive persistence
```

Hook targets live in `config.py` as dotted paths resolved by
`model_utils.get_module`, so a transformers upgrade that renames a module
fails loudly in one place instead of silently capturing the wrong tensor.

## Setup

Needs `transformers>=5.11.0` (DiffusionGemma was added there) and a CUDA build
of torch matching the host driver — see `requirements.txt`. On Colab use
`notebooks/run_all.ipynb`, which handles Drive persistence, the torch/torchaudio
version conflict, and HF authentication.

Model weights are gated; authenticate with an HF token that has access before
step 0.
