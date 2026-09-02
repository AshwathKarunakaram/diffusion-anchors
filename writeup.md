# Answer lock-in in DiffusionGemma

**When DiffusionGemma commits to a wrong answer early, whether it can still fix
itself is carried in its self-conditioning channel.** Transplanting that channel
from a run that self-corrected rescues 91% of stuck runs; transplanting it from
a *different stuck run* rescues 20%; scrambling it rescues 33%.

## Background

DiffusionGemma does not write left to right. It fills a 256-token canvas over
repeated denoising steps, freezing tokens it is confident about and
re-randomising the rest, so every token stays revisable until the end. It also
feeds its previous step's output distribution back into itself each step through
a **self-conditioning** channel — a recurrent state separate from the visible
canvas, with no autoregressive analogue. That channel is the object of study
here.

Prompts are short integer-answer maths questions with "State your answer first,
then give your reasoning", which forces early commitment to answer tokens while
the reasoning fills in around them. Same prompt, different random seeds, every
intermediate canvas recorded. Single canvas, inference only, one A100, no
finetuning.

## 1. The phenomenon

250 trajectories over 19 prompt families. Four families produce both fates on
the *same prompt*: some seeds write a wrong answer and later fix it, others
write one and never do — while the reasoning printed beside it is correct and
denoising steps remain.

| family | correct | locked | self-corrected |
|---|---|---|---|
| two_aces_hand | 32/40 | 6 | 32 |
| increasing_three_digits | 36/40 | 4 | 32 |
| one_ace_hand | 0/10 | 10 | 0 |
| squares_400_800 | 0/10 | 10 | 0 |

Wrong answers are **attractors, not noise**: the same wrong value recurs across
independent seeds. On `increasing_three_digits` every wrong run returns 120
where gold is 84 — and 120 = C(10,3) against 84 = C(9,3), i.e. the specific
error of allowing 0 as a leading digit. The stable mistake corresponds to a
nameable wrong method, not a slip.

Correction is close to one-way. Across 250 runs, wrong→right self-correction
happened over 100 times; right→wrong abandonment happened **4 times (1.6%)**.

## 2. The causal result

Take the self-conditioning state from a run that self-corrected, transplant it
into a stuck run at an early denoising step, continue generation. The natural
correct rate for this prompt is 80%; unpatched recipients are 0% by
construction, verified by replay before every patch.

![Rescue rate by condition and region, against the 80% natural correct rate](results/plots/patch/conditions_two_aces_hand.png)

| intervention | region | rescued | mean relative change to S |
|---|---|---|---|
| corrector's S | whole canvas | **48/53 (91%)** | 0.73 |
| another stuck run's S | whole canvas | 4/20 (20%) | 0.56 |
| shuffled S | whole canvas | 12/36 (33%) | 1.04 |

Corrector vs stuck-donor **p = 1.3×10⁻⁸**; corrector vs shuffled
**p = 2.0×10⁻⁸**; corrector vs the 80% baseline p = 0.032.

Three alternative explanations are ruled out:

- **Nondeterminism** — every recipient's unpatched replay must reproduce its
  stuck answer, or the pair is discarded rather than counted.
- **Any large perturbation frees the lock** — shuffled S makes the *largest*
  edit (1.04 relative) and rescues least.
- **Any coherent state resets the computation** — a stuck run's S is perfectly
  coherent and rescues at 20%.

Edit size does not order the outcomes; content does.

## 3. Where and when the outcome is decided

**The outcome lives in the answer positions.** Patching everything *except* the
answer window rescues 6/17 (35%), below the 80% baseline — patching around the
answer is merely disruptive.

**There is no cliff.**

![Rescue rate against the denoising step at which the patch is applied](results/plots/patch/pnr_curve_two_aces_hand.png)

| patch step | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| rescued | 89% | 81% | 67% | 58% | 50% |

A gradual decay. The outcome consolidates progressively rather than becoming
fixed at one step, and remains partially editable well past the point where the
model itself has stopped revising.

## 4. What failed

**A single direction does not explain it.** Difference-of-means steering
(correcting minus stuck, in self-conditioning hidden space) rescued 36/50 versus
25/50 for a norm-matched random direction — p = 0.04, and the 72% rescue rate
sits *below* the 80% natural rate, so it does not beat a reroll. Whole-state
transplant works at 91%; a rank-1 summary of the same signal does not. The
signal is causally sufficient but not low-rank.

**A linear probe cannot predict the outcome.** Logistic probes on step-1
self-conditioning state, leave-one-run-out: AUC 0.46 and 0.49 on one family,
0.04 and 0.65 on the other; cross-prompt transfer 0.35–0.66. All chance.

**Expert routing shows nothing.** Between-group Jensen-Shannon divergence over
MoE router distributions never exceeded the within-group baseline at any of 30
layers.

**Logit lens is blind exactly where it would be most useful.**

![Logit-lens top-1 agreement across decoder layer and denoising step](results/plots/lens/heatmap_agree_step.png)

Layer 29 agrees with the model's real output at 1.000, which validates the
instrument; layer 15 agrees at ≤0.008. In the unreadable band the answer region
decodes to cross-lingual number words — "eighty", "vingt", "nine" — suggesting
pre-lexical magnitude rather than digit tokens, which is plausibly *why* a fixed
unembedding cannot read it. A tuned lens is the obvious next instrument and was
out of scope here.

## 5. Limitations

- **Pseudo-replication.** The patch counts (n = 53 and similar) come from ~6
  independent stuck recipients measured across donors, steps, and regions. The
  exact tests treat rows as independent and therefore overstate significance.
  The effect size is large and the ordering is consistent across every cut, but
  the p-values should be read as descriptive rather than inferential.
- **Cross-session nondeterminism.** The MoE path uses `torch.histc`, which has
  no deterministic CUDA kernel, so identical seeds do not reproduce exactly
  across sessions; this family's natural rate moved 72.5% → 80% between runs.
  Patching validity is preserved because every replay was verified
  within-session.
- **Magnitudes not perfectly matched.** Corrector patches change S by 0.73
  relative, stuck-donor patches by 0.56. Shuffle's larger edit and worse outcome
  argues against size driving the result, but the gap remains.
- Single model, short integer-answer prompts, single canvas. The "internally
  absent" claims hold only within the lens-readable band (layers ≈26–29).

## 6. Directions killed before this one

**Code repair via variable renaming.** Renamed a definition mid-denoising and
watched whether dependents updated. The model deterministically either kept or
reverted the name according to a lexical prior, with no gradual repair dynamics.
A consistent rename is not a real inconsistency — there was nothing to repair.
Killed in a day.

**Predicting hidden code failure.** Screened 12 short Python tasks for
same-prompt pass/fail variation, intending to train a correctness probe on
intermediate states. Ten of twelve passed every time, and the failures were
syntax errors already visible on the canvas. No subtle-wrong-answer class
existed to probe. Killed in a day.

Both screens ran before anything expensive was built, against criteria fixed in
advance.

## Reproducing

```bash
git clone -b final_clean https://github.com/AshwathKarunakaram/diffusion-anchors
```

`README.md` lists the pipeline in order. The two long GPU stages take `--resume`
and skip cached work. `notebooks/run_all.ipynb` runs the whole thing on Colab
with Drive persistence.
