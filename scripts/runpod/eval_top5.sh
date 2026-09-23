#!/usr/bin/env bash
# Final evaluation of the top-5 FastEnhancer-T checkpoints on the VoiceBank-DEMAND
# testset (DNSMOS, SCOREQ, SI-SNR, PESQ, STOI, ESTOI, Whisper WER).
# Results land in /workspace/results/fastenhancer_t_vbd16k.
set -euo pipefail
repo=/workspace/fastenhancer
data=/workspace/data/voicebank-demand
name=vbd/16khz/fastenhancer_t
out=/workspace/results/fastenhancer_t_vbd16k

cd "$repo"
source .venv/bin/activate
python -m pip install -q onnxruntime openai-whisper jiwer requests
transcript="$data/logfiles/transcript_testset.txt"
[[ -f $transcript ]] || curl -fsSL -o "$transcript" \
  https://github.com/aask1357/fastenhancer/releases/download/test-data-v1/transcript_testset.txt

python -m scripts.runpod.eval_top5 -n "$name" --transcript-dir "$transcript" --out "$out" "$@"
