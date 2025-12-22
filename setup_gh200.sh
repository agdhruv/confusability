echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "Removing existing .venv directory..."
rm -rf .venv # this takes a while on a GH200 node
echo "Running uv sync..."
uv sync # also takes a while
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Uninstalling torch..."
uv pip uninstall torch
echo "Installing torch and torchvision with CUDA 12.4..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install datasets transformers trl lm-eval
echo "Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "Setup complete!"