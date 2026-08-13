"""Central config for the answer-anchoring project."""

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# --- Data ---
N_PROBLEMS = 50            # start small; scale after pipeline works
MAX_GOLD_STEPS = 3         # only GSM8K problems with short gold solutions
MAX_SOLUTION_CHARS = 400   # solution must fit comfortably in one 256-token canvas

# --- Generation ---
CANVAS_LENGTH = 256
MAX_DENOISING_STEPS = 48   # model default; keep explicit so it's logged

# --- Intervention ---
# Injection points as fractions of the interval [answer_commit_step, convergence_step]
INJECTION_FRACTIONS = [0.25, 0.50, 0.75]
# Matched wrong answers: same digit count, small perturbation. Tried in order.
ANSWER_DELTAS = [+2, -2, +10, -10, +1]

# --- Paths ---
TRAJ_DIR = "data/trajectories"       # cached per-step canvases
INTERV_DIR = "results/interventions"
PLOTS_DIR = "results/plots"

# --- Commitment definitions ---
# Answer is "committed" at the first step where the decoded answer span equals
# the final answer AND stays equal for all subsequent steps (stability, not
# first appearance -- first appearance overestimates commitment).
# Reasoning is "converged" at the first step from which >= REASONING_MATCH_FRAC
# of reasoning-region tokens match the final canvas at every later step.
REASONING_MATCH_FRAC = 0.95
