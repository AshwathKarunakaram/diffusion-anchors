"""Gate 1 deliverable: an instrumented copy of DiffusionGemma's denoising loop.

WHY THIS FILE EXISTS
`model.generate()` cannot inject an edited canvas mid-trajectory (see
CONTEXT.md / README "decoder_input_ids is NOT sound for mid-trajectory
injection" -- it resets `cur_step` to `max_denoising_steps`, so the
temperature schedule restarts). This file re-drives the SAME per-step
function the library uses, one step at a time, so we can pause between
steps and edit the canvas (o^t) and/or the self-conditioning logits (S^t)
without resetting anything.

DESIGN DECISION -- READ BEFORE EDITING
Instead of retyping the step math from `_denoising_step`
(generation_diffusion_gemma.py:1003-1076) by hand, this module calls that
exact private method, once per step, in the exact order `generate()` calls
it. The only new code here is the outer loop and the hook between steps.
This means `intervention_fn=None` does not merely "resemble" HF generation
-- it structurally IS HF generation for a single canvas block, because
there is no reimplementation of the math to drift out of sync. The
trade-off: this relies on transformers-internal private methods
(`_prepare_denoiser_inputs`, `_denoising_step`, etc.), so it can break on a
transformers version bump. Re-run the parity check below after any
`pip install -U transformers`.

SCOPE (deliberately minimal -- do not extend without a reason)
  - Single canvas block only. Every problem in this project fits in one
    256-token block (config.py: CANVAS_LENGTH=256, MAX_SOLUTION_CHARS=400).
    Raises NotImplementedError if a second block would be needed.
  - batch_size == 1 only (same constraint CanvasRecorder already has).
  - No save/load of loop state to disk. `intervention_fn` runs inside one
    live GPU call; if you need to inject at step k, just run this function
    once and check `cur_step == k` inside your hook -- there is no need to
    serialize partial state.
"""

import math

import torch


class StepRecord(dict):
    """One denoising step's log entry. Plain dict subclass so callers can
    index it (`step["cur_step"]`) or json.dumps a stripped copy freely."""


def _entropy_stats(logits: torch.Tensor) -> tuple[float, float]:
    """(mean, max) token-level entropy, matching the distribution the
    sampler itself scores (`EntropyBoundSampler.accept_canvas`, generation_
    diffusion_gemma.py:437-438: `Categorical(logits=logits).entropy()`)."""
    dist = torch.distributions.Categorical(logits=logits.float())
    ent = dist.entropy()  # (batch, canvas_length)
    return ent.mean().item(), ent.max().item()


@torch.no_grad()
def run_denoising(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor | None = None,
    generation_config=None,
    intervention_fn=None,
    disable_compile: bool = True,
    log_logits: bool = False,
    seed: int | None = None,
):
    """Runs one canvas block of DiffusionGemma denoising, step by step.

    Args:
        model: a loaded `DiffusionGemmaForBlockDiffusion`.
        input_ids: prompt token ids, shape (1, prompt_len).
        attention_mask: optional (1, prompt_len) bool/int mask. Defaults to
            all-ones, same as `generate()` (generation_diffusion_gemma.py:694).
        generation_config: a `DiffusionGemmaGenerationConfig`, or None to use
            the model's default (matches `generate()`'s own resolution order).
        intervention_fn: `None`, or a callable `(state: dict) -> dict | None`
            called once per step, AFTER `_denoising_step` has produced the
            canvas/self-conditioning that will feed the NEXT step. `state`
            contains (all on `model.device` unless noted):
              cur_step            -- the step that just completed (int,
                                      counts DOWN, same convention as HF)
              current_canvas      -- (1, canvas_length) LongTensor, about to
                                      be fed into the next decoder call (o^t)
              argmax_canvas       -- (1, canvas_length) LongTensor, this
                                      step's "best guess" (what the streamer
                                      shows)
              self_conditioning_logits -- (1, canvas_length, vocab) tensor,
                                      about to condition the next step (S^t)
              accepted_mask       -- (1, canvas_length) BoolTensor, which
                                      positions the sampler accepted this step
              temperature         -- float, this step's schedule temperature
            To intervene, return a dict with any of `current_canvas` /
            `self_conditioning_logits` set to replacement tensors of the same
            shape/dtype/device; omitted keys are left untouched. Returning
            `None` (or not passing `intervention_fn`) changes nothing, which
            is why `intervention_fn=None` reproduces `generate()` exactly.
        disable_compile: forces the eager (uncompiled) decoder forward, same
            as passing `disable_compile=True` to `generate()`. Keep this
            True on both sides of any parity comparison -- torch.compile's
            cudagraph capture interacts with the sampler's RNG draws in a
            way that is not guaranteed to line up step-for-step outside the
            graph (see `torch.compiler.cudagraph_mark_step_begin()` at
            generation_diffusion_gemma.py:1025, "needed for compiled EB
            sampler").
        log_logits: if True, also stores the full (canvas_length, vocab)
            self-conditioning logits tensor (CPU, per step) in the log. Off
            by default -- vocab_size is 262144, so this is ~134MB/step in
            bf16 and will blow through host RAM over 48 steps if left on for
            a full run. Turn on only when diffing a couple of steps.
        seed: if given, calls `torch.manual_seed` / `torch.cuda.manual_seed_all`
            right before the loop starts. Pass the SAME seed to both the HF
            and custom runs in a parity check -- see note on RNG below.

    Returns:
        (final_argmax_canvas: LongTensor (1, canvas_length), steps: list[StepRecord])

    RNG NOTE (read before trusting a parity diff)
    Every stochastic op in the loop -- `sampler.initialize_canvas`'s
    `torch.randint` (generation_diffusion_gemma.py:398-403) and
    `_denoising_step`'s `torch.multinomial` (line 1045) -- draws from
    PyTorch's global default RNG on the model's device. No `generator=` is
    ever passed (confirmed: zero hits for "generator" in
    generation_diffusion_gemma.py and modeling_diffusion_gemma.py). That
    means:
      1. The two runs must be seeded identically immediately before they
         start, with nothing else touching the RNG in between.
      2. Because this function calls HF's own `_prepare_denoiser_inputs`
         and `_denoising_step` verbatim, it draws random numbers in the
         exact same order, same shapes, as `generate()` does -- there is no
         extra draw introduced by this file to cause a desync.
      3. A SEPARATE risk from RNG: GPU matmul/attention kernels can be
         numerically non-deterministic between runs (parallel reduction
         order) even with identical seeds and identical weights. This can
         flip an argmax or entropy comparison right at a tie, which LOOKS
         like a logic bug but isn't. If you see a divergence, first retry
         with `torch.use_deterministic_algorithms(True)` (set once, at
         process start) before concluding the loop logic is wrong.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = input_ids.device
    batch_size, cur_len = input_ids.shape
    if batch_size != 1:
        raise NotImplementedError("run_denoising only supports batch_size == 1.")

    generation_config, model_kwargs = model._prepare_generation_config(
        generation_config, disable_compile=disable_compile, max_new_tokens=model.config.canvas_length
    )
    if model_kwargs:
        raise NotImplementedError(f"Unhandled generate() kwargs: {list(model_kwargs)}")

    max_length, max_new_tokens = model._prepare_generated_length(generation_config, cur_len)
    canvas_length = model.config.canvas_length
    max_new_canvases = math.ceil(max_new_tokens / canvas_length)
    if max_new_canvases != 1:
        raise NotImplementedError(
            f"run_denoising is single-block only, but max_new_tokens={max_new_tokens} needs "
            f"{max_new_canvases} blocks. Pass max_new_tokens<={canvas_length} via generation_config."
        )

    past_key_values = model._prepare_cache_for_generation(
        generation_config=generation_config, batch_size=batch_size, max_length=max_length
    )

    encoder_position_ids = torch.arange(0, cur_len, dtype=torch.int32, device=device).unsqueeze(0)
    decoder_position_ids = torch.arange(cur_len, cur_len + canvas_length, dtype=torch.int32, device=device).unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, cur_len), dtype=torch.bool, device=device)
    else:
        attention_mask = attention_mask.bool()
    decoder_attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)

    sampler = model._prepare_sampler(generation_config)
    logits_processor = model._prepare_logits_processor(generation_config, None)
    diffusion_stopping_criteria = model._prepare_diffusion_stopping_criteria(generation_config)

    decoder_forward = model.forward  # eager; `disable_compile=True` above keeps `generate()` on this same path

    # 1.a Encode the prompt (prefill) -- identical to generate()'s first outer-loop iteration.
    unprocessed_input_ids, encoder_mask_mapping = model._prepare_encoder_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_position_ids=encoder_position_ids,
        past_key_values=past_key_values,
        is_prefill=True,
        canvas_length=canvas_length,
        batch_size=batch_size,
    )
    encoder_outputs = model.model.encoder(
        input_ids=unprocessed_input_ids,
        attention_mask=encoder_mask_mapping,
        past_key_values=past_key_values,
        position_ids=encoder_position_ids,
    )
    past_key_values = encoder_outputs.past_key_values

    # 1.b Prepare the denoising loop's starting canvas / self-conditioning / mask.
    current_canvas, self_conditioning_logits, mask_mapping, finished_denoising = model._prepare_denoiser_inputs(
        decoder_attention_mask=decoder_attention_mask,
        past_key_values=past_key_values,
        sampler=sampler,
        diffusion_stopping_criteria=diffusion_stopping_criteria,
        batch_size=batch_size,
        device=device,
        model_kwargs={},
    )
    argmax_canvas = current_canvas

    steps = []
    for cur_step in reversed(range(1, generation_config.max_denoising_steps + 1)):
        current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = model._denoising_step(
            decoder_forward=decoder_forward,
            current_canvas=current_canvas,
            argmax_canvas=argmax_canvas,
            input_ids=input_ids,
            decoder_position_ids=decoder_position_ids,
            self_conditioning_logits=self_conditioning_logits,
            mask_mapping=mask_mapping,
            past_key_values=past_key_values,
            finished_denoising=finished_denoising,
            cur_step=cur_step,
            sampler=sampler,
            logits_processor=logits_processor,
            diffusion_stopping_criteria=diffusion_stopping_criteria,
        )

        temperature = generation_config.t_min + (
            (generation_config.t_max - generation_config.t_min) * (cur_step / generation_config.max_denoising_steps)
        )
        mean_ent, max_ent = _entropy_stats(self_conditioning_logits)
        record = StepRecord(
            cur_step=int(cur_step),
            temperature=float(temperature),
            argmax_canvas=argmax_canvas[0].tolist(),
            accepted_mask=sampler.accepted_token_mask[0].tolist(),
            num_accepted=int(sampler.accepted_token_mask.sum().item()),
            mean_entropy=mean_ent,
            max_entropy=max_ent,
            finished=bool(finished_denoising[0].item()),
        )
        if log_logits:
            record["self_conditioning_logits"] = self_conditioning_logits[0].cpu()

        if intervention_fn is not None:
            state = {
                "cur_step": int(cur_step),
                "current_canvas": current_canvas,
                "argmax_canvas": argmax_canvas,
                "self_conditioning_logits": self_conditioning_logits,
                "accepted_mask": sampler.accepted_token_mask,
                "temperature": temperature,
            }
            overrides = intervention_fn(state)
            if overrides:
                if "current_canvas" in overrides:
                    current_canvas = overrides["current_canvas"]
                    record["intervened_canvas"] = True
                if "self_conditioning_logits" in overrides:
                    self_conditioning_logits = overrides["self_conditioning_logits"]
                    record["intervened_self_conditioning"] = True

        steps.append(record)
        if torch.all(finished_denoising):
            break

    return argmax_canvas, steps


if __name__ == "__main__":
    import sys

    from lockin_prompts import PROMPTS
    from model_utils import build_chat_inputs, load_model, CanvasRecorder

    print("=== Gate 1 parity check: HF generate() vs custom_denoise.run_denoising ===")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("torch.use_deterministic_algorithms(True, warn_only=True): OK")
    except Exception as e:
        print(f"torch.use_deterministic_algorithms not supported here: {e}\n"
              "Continuing anyway -- if we see a mismatch, this is the first thing to suspect.")
    print("NOTE: this is a 26B MoE model -- its expert-routing path "
          "(transformers/integrations/moe.py:grouped_mm_experts_forward) uses torch.histc, "
          "which has NO deterministic CUDA kernel. That means even two back-to-back HF "
          "generate() calls with the same seed can differ by GPU floating-point noise. "
          "warn_only=True lets it run anyway; any divergence found below may be this, not our loop.")

    model, processor = load_model()
    prompt = PROMPTS[0]
    inputs = build_chat_inputs(processor, prompt.prompt, model.device)
    SEED = 0

    # --- run A: stock HF generate() ---
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    rec = CanvasRecorder(tokenizer=processor.tokenizer)
    hf_out = model.generate(**inputs, max_new_tokens=256, streamer=rec, disable_compile=True)
    hf_final = hf_out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
    hf_drafts = rec.draft_history  # list[list[int]], one per step

    # --- run B: our loop, intervention_fn=None ---
    final_canvas, steps = run_denoising(
        model, inputs["input_ids"], inputs.get("attention_mask"), disable_compile=True, seed=SEED
    )
    custom_final = final_canvas[0].tolist()
    custom_drafts = [s["argmax_canvas"] for s in steps]

    print(f"HF steps: {len(hf_drafts)}   custom steps: {len(custom_drafts)}")

    divergence = None
    for i, (hf_ids, custom_ids) in enumerate(zip(hf_drafts, custom_drafts)):
        if hf_ids != custom_ids:
            first_pos = next(j for j in range(len(hf_ids)) if hf_ids[j] != custom_ids[j])
            divergence = (i, first_pos)
            break

    # `_finalize_canvas` (generation_diffusion_gemma.py, called only by generate()) pads
    # everything after the first EOS with `pad_token_id`. This loop deliberately doesn't
    # implement that (it's the multi-block/finished-sequence bookkeeping this file is scoped
    # to skip -- see module docstring), so trailing filler can legitimately differ. Truncate
    # both sequences at the first EOS before the final comparison.
    eos_ids = set(model.generation_config.eos_token_id or [])
    def _truncate_at_eos(ids):
        for i, t in enumerate(ids):
            if t in eos_ids:
                return ids[: i + 1]
        return ids
    hf_content = _truncate_at_eos(hf_final)
    custom_content = _truncate_at_eos(custom_final)

    if divergence is None and hf_content == custom_content:
        print("PARITY OK: every step's argmax canvas matches, and content up to the first "
              "EOS token matches exactly. (Trailing post-EOS filler may differ -- that's "
              "`_finalize_canvas`'s pad-replacement, which this minimal loop doesn't run; "
              "see module docstring.)")
    else:
        print("PARITY MISMATCH.")
        if divergence is not None:
            step_i, pos = divergence
            rec_at_step = steps[step_i]
            print(f"  first divergent denoising step (0-indexed into the log): {step_i} "
                  f"(cur_step={rec_at_step['cur_step']})")
            print(f"  first divergent canvas position: {pos}")
            print(f"  HF token id there:     {hf_drafts[step_i][pos]}")
            print(f"  custom token id there: {custom_drafts[step_i][pos]}")
            print(f"  temperature at this step: {rec_at_step['temperature']:.4f}")
            print(f"  custom accepted-mask at this position: {rec_at_step['accepted_mask'][pos]}")
            print(f"  custom mean/max entropy this step: {rec_at_step['mean_entropy']:.4f} / "
                  f"{rec_at_step['max_entropy']:.4f}")
            print("  NOTE: if this is the ONLY thing that differs (i.e. re-running this exact "
                  "script twice gives a DIFFERENT divergence point or position each time), that's "
                  "GPU kernel non-determinism, not an RNG-stream desync or a logic bug -- stop and "
                  "report that, per the parity validation plan, rather than debugging the loop further.")
        else:
            print(f"  all logged steps matched, but final decoded sequences differ.")
            print(f"  HF final ids:     {hf_final}")
            print(f"  custom final ids: {custom_final}")
