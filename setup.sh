#!/usr/bin/env bash
# One-shot setup for a fresh RunPod GPU rental. Assumes your persistent
# network volume (with cached HF weights under /workspace/hf) is already
# attached -- this script does NOT download the 26B model. It only installs
# OS packages, Python dependencies, and clones the repo.
#
# Usage (on the fresh pod, inside tmux):
#   curl -fsSL <raw-url-to-this-file> -o setup.sh && bash setup.sh
#   # or, if you cloned manually first: bash diffusion-anchors/setup.sh
#
# Idempotent: safe to re-run if something fails partway through.

set -euo pipefail

REPO_URL="https://github.com/AshwathKarunakaram/diffusion-anchors.git"
REPO_DIR="/workspace/diffusion-anchors"
BRANCH="initial_scaffolding"   # NOT main -- every fix so far lives here, unmerged

echo "=== 1/6: apt packages ==="
apt-get update -qq
apt-get install -y -qq tmux git

echo "=== 2/6: HF_HOME -> persistent volume ==="
export HF_HOME=/workspace/hf
if ! grep -qxF 'export HF_HOME=/workspace/hf' ~/.bashrc 2>/dev/null; then
    echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc
fi
if [ -d "$HF_HOME/hub/models--google--diffusiongemma-26B-A4B-it" ]; then
    echo "Found cached weights at \$HF_HOME -- good, nothing to download."
else
    echo "WARNING: no cached weights found at $HF_HOME/hub -- if your network"
    echo "volume didn't actually attach, the first script that loads the model"
    echo "will download ~49GB. Check your volume mount before proceeding."
fi

echo "=== 3/6: clone / update repo (branch: $BRANCH) ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== 4/6: python dependencies ==="
python3 -m pip install --upgrade pip -q
# RunPod's PyTorch template ships a torchaudio pinned to whatever torch the
# image started with; installing a different torch below leaves torchaudio
# import-broken unless it's removed first.
python3 -m pip uninstall -y torchaudio -q || true
python3 -m pip install -r requirements.txt

echo "=== 5/6: sanity-check the install ==="
python3 -c "
import torch
print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
from transformers import DiffusionGemmaForBlockDiffusion
print('DiffusionGemmaForBlockDiffusion import: OK')
"

echo "=== 6/6: credentials reminder ==="
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" >/dev/null
    echo "Logged into HF with \$HF_TOKEN."
else
    echo "No \$HF_TOKEN set. If the volume's weights are gated/missing, run:"
    echo "  huggingface-cli login"
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Reminder: judge.py (step 4) needs ANTHROPIC_API_KEY exported -- not set yet."
fi

cat <<'EOF'

=== Setup done. Next steps ===

  tmux new -s main
  export HF_HOME=/workspace/hf
  export CUBLAS_WORKSPACE_CONFIG=:4096:8   # only needed if you re-run custom_denoise.py's parity check
  cd /workspace/diffusion-anchors

  python src/smoke_test.py                              # step 0, ~2min, sanity check
  python src/generate_trajectories.py 2>&1 | tee data/gen.log        # step 1, GPU
  python src/parse_commitment.py                                    # step 2, CPU
  python src/intervene_swap.py 2>&1 | tee results/intervene.log     # step 3, GPU
  export ANTHROPIC_API_KEY=...
  python src/judge.py 2>&1 | tee results/judge.log                  # step 4, CPU + API

See README.md's "Verified-assumptions checklist" and CONTEXT.md before
trusting any of this blindly -- both log exactly what's been fixed and
what's still known-broken (problem selection bias, missing AR-Gemma
baseline, missing plotting script).
EOF
