#!/usr/bin/env bash
# Full pipeline on a RunPod GPU pod: env setup, VoiceBank-DEMAND 16 kHz prep,
# upstream test run, then FastEnhancer-T training (batch 64, 2 s clips).
# Idempotent: finished stages are skipped on re-run. Run under nohup.
set -euo pipefail
repo=/workspace/fastenhancer
data=/workspace/data/voicebank-demand
here="$(cd "$(dirname "$0")" && pwd)"

stage() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

stage "1/5 environment"
if [[ ! -f /workspace/.setup_done ]]; then
  bash "$here/setup.sh"
  touch /workspace/.setup_done
fi
source "$repo/.venv/bin/activate"
python -m pip install -q soundfile soxr

stage "2/5 download VoiceBank-DEMAND (48 kHz, 28 speakers)"
api=https://datashare.ed.ac.uk/server/api/core/bitstreams
declare -A files=(
  [clean_testset_wav]=dec213d3-bf57-4777-9663-c24bdce92d5e
  [noisy_testset_wav]=13c1bfbf-14a6-41db-9b41-8f7310f01ad5
  [clean_trainset_28spk_wav]=245452b6-6235-44b6-a6f9-e7eb19797769
  [noisy_trainset_28spk_wav]=ecb5a102-bb00-46d3-8af5-40c79823b837
  [logfiles]=11185dc8-9cf1-405b-b858-35bd6a04aedd
)
mkdir -p "$data/zips" "$data/48k"
for name in "${!files[@]}"; do
  [[ -f "$data/zips/$name.done" ]] && continue
  curl -fL --retry 5 -C - -o "$data/zips/$name.zip" "$api/${files[$name]}/content"
  if [[ $name == logfiles ]]; then
    unzip -qo "$data/zips/$name.zip" -d "$data/logfiles" -x '__MACOSX/*'
  else
    unzip -qo "$data/zips/$name.zip" -d "$data/48k" -x '__MACOSX/*'
  fi
  rm "$data/zips/$name.zip"
  touch "$data/zips/$name.done"
done
for d in clean_testset_wav noisy_testset_wav clean_trainset_28spk_wav noisy_trainset_28spk_wav; do
  echo "$d: $(ls "$data/48k/$d" | wc -l) files"
done
ls "$data/logfiles"

stage "3/5 resample to 16 kHz"
cd "$repo"
if [[ ! -f "$data/16k.done" ]]; then
  python -m scripts.resample --to-sr 16000 --from-dir "$data/48k" --to-dir "$data/16k" --num-workers "$(nproc)"
  touch "$data/16k.done"
  rm -rf "$data/48k"
fi

sed "s#/home/shahn/Datasets/voicebank-demand#$data#g" configs/fastenhancer/t.yaml > configs/fastenhancer/t_runpod.yaml

stage "4/5 upstream test run (10 steps + valid + metrics)"
if [[ ! -f /workspace/.test_done ]]; then
  python train.py -n delete_it -c configs/fastenhancer/t_runpod.yaml \
    -p train.test=True pesq.interval=1 -f
  rm -rf logs/delete_it
  touch /workspace/.test_done
fi

stage "5/5 training FastEnhancer-T"
name=vbd/16khz/fastenhancer_t
if [[ -f "logs/$name/config.yaml" ]]; then
  python train.py -n "$name"   # resume from latest checkpoint
else
  python train.py -n "$name" -c configs/fastenhancer/t_runpod.yaml -f
fi
