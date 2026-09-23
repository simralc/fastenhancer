#!/usr/bin/env bash
# Run on a Linux NVIDIA GPU pod with Python 3.11-3.13 and git available.
set -euo pipefail
task_root="${FASTENHANCER_ROOT:-/workspace/fastenhancer}"
if [[ ! -d "$task_root/.git" ]]; then
  git clone -b runpod-training https://github.com/simralc/fastenhancer.git "$task_root"
fi
cd "$task_root"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install matplotlib tensorboard scipy librosa unidecode einops cython tqdm pyyaml pesq pystoi torch-pesq torchmetrics
python -m pip check
python - <<'PY'
import torch
import yaml
from models.fastenhancer.default.model import Model
from wrappers.ns import ModelWrapper
assert torch.cuda.is_available(), 'A CUDA-capable pod is required'
print('GPU:', torch.cuda.get_device_name(0))
with open('configs/fastenhancer/t.yaml') as f:
    config = yaml.safe_load(f)
model = Model(**config['model_kwargs']).cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for step in range(3):
    noisy = torch.randn(2, 32000, device='cuda') * 0.02
    enhanced, _ = model(noisy)
    loss = (enhanced - noisy[:, :enhanced.shape[-1]]).square().mean()
    assert torch.isfinite(loss), loss
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    print('Synthetic smoke step:', step, 'loss:', loss.item())
print('Peak allocated GPU memory (MiB):', torch.cuda.max_memory_allocated() / 2**20)
print('GPU forward/backward smoke test passed; this is not dataset training.')
PY
python -m pip freeze > /workspace/fastenhancer-environment.txt
