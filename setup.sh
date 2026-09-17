#!/usr/bin/env bash
# setup.sh -- create/update the `panoptex` conda env for the perception stack,
# and download the GroundingDINO/SAM2 model weights this repo needs.
#
#   ./setup.sh                 auto-detect CUDA toolkit, install matching torch
#   ./setup.sh --cpu-only      skip CUDA detection, install CPU-only torch
#   ./setup.sh --recreate      drop and recreate the panoptex env first
#   ./setup.sh --src-dir PATH  editable checkout location (default: $CONDA_PREFIX/src)
#
# Safe to re-run.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "setup.sh needs bash. Run: bash setup.sh" >&2
  exit 1
fi

set -euo pipefail

info() { printf '[setup.sh] %s\n' "$*"; }
warn() { printf '[setup.sh] WARNING: %s\n' "$*" >&2; }
die()  { printf '[setup.sh] ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f requirements.txt ] || die "requirements.txt not found in $SCRIPT_DIR -- run this from the repo root."
[ -f requirements-models.txt ] || die "requirements-models.txt not found in $SCRIPT_DIR -- run this from the repo root."

ENV_NAME=panoptex
PY_VER=3.10
TORCH_VERSION=2.5.1
TORCHVISION_VERSION=0.20.1
# CUDA tags torch==${TORCH_VERSION}/torchvision==${TORCHVISION_VERSION} ship official wheels for.
SUPPORTED_CUDA=(11.8 12.1 12.4)

RECREATE=0
CPU_ONLY=0
SRC_DIR_OVERRIDE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --recreate) RECREATE=1; shift ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --src-dir)
      [ $# -ge 2 ] || die "--src-dir requires a path argument"
      SRC_DIR_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "Unknown argument: $1 (see --help)" ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. conda present?
# ---------------------------------------------------------------------------
command -v conda >/dev/null 2>&1 || die \
  "conda not found on PATH. Install Miniconda or Miniforge first: https://docs.conda.io/en/latest/miniconda.html, then re-run ./setup.sh."

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. create / reuse the env
# ---------------------------------------------------------------------------
env_exists() { conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; }

if [ "$RECREATE" -eq 1 ] && env_exists; then
  info "removing existing '$ENV_NAME' env (--recreate)"
  conda env remove -n "$ENV_NAME" -y
fi

if env_exists; then
  info "reusing existing '$ENV_NAME' env"
  existing_py="$(conda run -n "$ENV_NAME" python --version 2>/dev/null | awk '{print $2}')"
  case "$existing_py" in
    "$PY_VER".*) ;;
    *) warn "'$ENV_NAME' has Python ${existing_py:-unknown}, expected ${PY_VER}.x. Re-run with --recreate to rebuild it." ;;
  esac
else
  info "creating '$ENV_NAME' env (python=$PY_VER)"
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi

# Must be set before activation to take effect. Fixes: this env otherwise
# silently picks up stray packages from ~/.local/lib/pythonX/site-packages.
conda env config vars set PYTHONNOUSERSITE=1 -n "$ENV_NAME" >/dev/null
conda activate "$ENV_NAME"

info "python: $(command -v python) ($(python --version 2>&1))"

# ---------------------------------------------------------------------------
# 3. soft ROS check (informational only -- not needed for this script)
# ---------------------------------------------------------------------------
if [ ! -f /opt/ros/humble/setup.bash ]; then
  warn "ROS 2 Humble not found at /opt/ros/humble/setup.bash. Not needed to" \
       "finish this script, but required later to build/run the ROS packages -- see README §1."
fi

# ---------------------------------------------------------------------------
# 4. base deps
# ---------------------------------------------------------------------------
info "installing base Python deps (requirements.txt)"
python -m pip install -q -U pip wheel
pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# 5. CUDA detection
# ---------------------------------------------------------------------------
find_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "$CUDA_HOME"; return 0
  fi
  # Prefer explicit /usr/local/cuda-* installs (NVIDIA repo layout) over
  # whatever `nvcc` happens to resolve first on PATH: distro packages like
  # Ubuntu's `nvidia-cuda-toolkit` put an older nvcc directly on PATH and
  # would otherwise shadow a newer toolkit installed correctly alongside it.
  local d
  for d in $(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tac); do
    if [ -x "${d}bin/nvcc" ]; then
      echo "${d%/}"; return 0
    fi
  done
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    echo /usr/local/cuda; return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    dirname "$(dirname "$(command -v nvcc)")"; return 0
  fi
  return 1
}

version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

map_cuda_tag() {
  local detected="$1" chosen="" v
  for v in "${SUPPORTED_CUDA[@]}"; do
    if version_ge "$detected" "$v"; then chosen="$v"; fi
  done
  if [ -z "$chosen" ]; then
    die "detected CUDA toolkit $detected predates every wheel torch==${TORCH_VERSION} ships (${SUPPORTED_CUDA[*]}) -- refusing to guess a mismatched build. Install a supported CUDA toolkit, or re-run: ./setup.sh --cpu-only"
  fi
  echo "cu${chosen/./}"
}

CUDA_TAG=""
CUDA_HOME_DETECTED=""

if [ "$CPU_ONLY" -eq 1 ]; then
  info "CPU-only mode requested (--cpu-only) -- skipping CUDA detection"
else
  if detected_home="$(find_cuda_home)"; then
    detected_ver="$(
      "$detected_home/bin/nvcc" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p'
    )"
    if [ -z "$detected_ver" ]; then
      warn "found nvcc at $detected_home but couldn't parse its version."
    else
      CUDA_HOME_DETECTED="$detected_home"
      CUDA_TAG="$(map_cuda_tag "$detected_ver")"
      info "detected CUDA toolkit $detected_ver at $detected_home -> using PyTorch wheel tag $CUDA_TAG"
    fi
  fi

  if [ -z "$CUDA_TAG" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      die "GPU detected but no usable CUDA toolkit/nvcc found (checked \$CUDA_HOME, PATH, /usr/local/cuda*). Either install a CUDA toolkit matching your driver, or re-run: ./setup.sh --cpu-only"
    else
      die "No NVIDIA GPU / CUDA toolkit detected. Re-run: ./setup.sh --cpu-only  to install a CPU-only build (perception nodes will run on CPU; slow, and GroundingDINO's compiled CUDA op won't be built)."
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 6. torch / torchvision
# ---------------------------------------------------------------------------
if [ -n "$CUDA_TAG" ]; then
  info "installing torch==${TORCH_VERSION}+${CUDA_TAG} torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}"
  pip install -q \
    "torch==${TORCH_VERSION}+${CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
  info "installing CPU-only torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}"
  pip install -q \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
fi

# ---------------------------------------------------------------------------
# 7. GroundingDINO + SAM2 (editable, need torch already installed)
# ---------------------------------------------------------------------------
SRC_DIR="${SRC_DIR_OVERRIDE:-$CONDA_PREFIX/src}"
mkdir -p "$SRC_DIR"
info "installing GroundingDINO + SAM2 (requirements-models.txt) into $SRC_DIR"

if [ -n "$CUDA_HOME_DETECTED" ]; then
  export CUDA_HOME="$CUDA_HOME_DETECTED"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

if ! pip install --no-build-isolation --src "$SRC_DIR" -r requirements-models.txt; then
  die "GroundingDINO/SAM2 install failed -- check network/GitHub access above. Safe to re-run: ./setup.sh"
fi

# ---------------------------------------------------------------------------
# 8. model weights -- downloaded straight into this repo (weights/, gitignored),
#    not left for the user to hunt down and clone somewhere under $HOME.
# ---------------------------------------------------------------------------
WEIGHTS_DIR="$SCRIPT_DIR/weights"
GDINO_WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GDINO_WEIGHTS_FILE="$WEIGHTS_DIR/groundingdino_swint_ogc.pth"
SAM2_WEIGHTS_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
SAM2_WEIGHTS_FILE="$WEIGHTS_DIR/sam2.1_hiera_small.pt"

info "checking model weights"
mkdir -p "$WEIGHTS_DIR"

if [ ! -f "$GDINO_WEIGHTS_FILE" ]; then
  info "downloading GroundingDINO weights -> $GDINO_WEIGHTS_FILE"
  wget -q --show-progress -O "$GDINO_WEIGHTS_FILE" "$GDINO_WEIGHTS_URL" \
    || die "failed to download $GDINO_WEIGHTS_URL"
else
  info "$GDINO_WEIGHTS_FILE already present, skipping download"
fi

if [ ! -f "$SAM2_WEIGHTS_FILE" ]; then
  info "downloading SAM2.1 (hiera_small) weights -> $SAM2_WEIGHTS_FILE"
  wget -q --show-progress -O "$SAM2_WEIGHTS_FILE" "$SAM2_WEIGHTS_URL" \
    || die "failed to download $SAM2_WEIGHTS_URL"
else
  info "$SAM2_WEIGHTS_FILE already present, skipping download"
fi

# ---------------------------------------------------------------------------
# 9. summary
# ---------------------------------------------------------------------------
info "verifying install..."
python - <<'PYEOF'
import torch
print(f"[setup.sh]   torch          : {torch.__version__} (cuda_available={torch.cuda.is_available()})")
try:
    from groundingdino import _C  # noqa: F401
    print("[setup.sh]   groundingdino _C: built (GPU ops available)")
except ImportError:
    print("[setup.sh]   groundingdino _C: not built -- detector will run on CPU")
import sam2  # noqa: F401
print("[setup.sh]   sam2           : importable")
import transformers
print(f"[setup.sh]   transformers   : {transformers.__version__} (must be 4.30.2 for GroundingDINO)")
PYEOF

# colcon must resolve to the copy inside this env, not /usr/bin/colcon --
# otherwise `colcon build` writes console_script shebangs for the system
# python and the nodes start outside the env.
COLCON_PATH="$(command -v colcon || true)"
case "$COLCON_PATH" in
  "$CONDA_PREFIX"/*) info "  colcon         : $COLCON_PATH" ;;
  "")                warn "colcon not on PATH after install -- 'conda activate $ENV_NAME' and re-check." ;;
  *)                 warn "colcon resolves to $COLCON_PATH, not this env. Open a fresh shell, activate '$ENV_NAME', and re-check -- building with the system colcon points the nodes at the wrong python." ;;
esac

cat <<EOF
[setup.sh]
[setup.sh] Done. Env: $ENV_NAME (python $PY_VER)
[setup.sh]   GroundingDINO config : $SRC_DIR/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py
[setup.sh]   GroundingDINO weights: $GDINO_WEIGHTS_FILE
[setup.sh]   SAM2 checkpoint      : $SAM2_WEIGHTS_FILE
[setup.sh] Before building or running ROS nodes, in this order:
[setup.sh]   source /opt/ros/humble/setup.bash
[setup.sh]   conda activate $ENV_NAME
EOF
