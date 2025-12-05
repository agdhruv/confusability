rm -rf .venv # this takes a while on a GH200 node
uv sync # also takes a while
source .venv/bin/activate
uv pip uninstall torch
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export CUDA_HOME=/usr/local/cuda
uv pip install flash-attn --no-build-isolation