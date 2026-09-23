# FastEnhancer-T on VoiceBank-DEMAND 16 kHz: training results

One training run of FastEnhancer-T using the upstream config (`configs/fastenhancer/t.yaml`, with the dataset paths rewritten). The five best checkpoints were evaluated in full with `scripts.metrics_ns`.

## Final evaluation (VoiceBank-DEMAND testset, 824 utterances)

| Rank | Epoch | DNSMOS P.808 | SIG | BAK | OVL | SCOREQ ↓ | SI-SDR (dB) | PESQ | STOI | ESTOI | WER (%) ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **500** | 3.40 | 3.32 | 4.01 | 3.04 | 0.354 | 18.14 | **2.89** | 0.937 | 0.846 | **3.48** |
| 2 | 480 | 3.40 | 3.32 | 4.00 | 3.04 | 0.356 | 18.14 | 2.87 | 0.937 | 0.846 | 3.50 |
| 3 | 460 | 3.40 | 3.32 | 4.00 | 3.04 | 0.359 | 18.09 | 2.86 | 0.937 | 0.846 | 3.69 |
| 4 | 440 | 3.40 | 3.32 | 4.01 | 3.05 | 0.360 | 18.18 | 2.84 | 0.937 | 0.845 | 3.70 |
| 5 | 400 | 3.40 | 3.31 | 4.03 | 3.04 | 0.363 | 17.76 | 2.83 | 0.933 | 0.840 | 4.08 |
| Paper (mean of 5 seeds) | 500 | 3.42 | 3.34 | 4.01 | 3.06 | 0.334 | 18.6 | 2.99 | 0.940 | 0.850 | 3.6 |

The paper row is Table 1 of the upstream [README](../../README.md). Full-precision values are in [`results.csv`](results.csv) and [`results.json`](results.json).

**Summary:** the final checkpoint (epoch 500) is the best on PESQ and WER. It is within 0.1 PESQ, 0.5 dB SI-SDR and 0.003 STOI of the paper's five-seed average, and its WER (3.48%) is slightly better. The gap fits the difference between one seed and a five-seed mean, but this single run can't confirm that.

## How checkpoints were selected

- A checkpoint is saved every 20 epochs, giving 25 candidates. Training also computes PESQ/STOI every 20 epochs (`metrics/pesq` in TensorBoard). The top 5 by that PESQ were evaluated.
- **Caveat:** VoiceBank-DEMAND has no separate dev set, and the upstream config validates on the testset. The checkpoints were therefore picked with the same testset they are scored on, so ranks 1–5 are slightly optimistic. Here the pick is also simply the last few epochs (400–500), where the learning rate has decayed. Epoch 500 would have been chosen with or without this selection step.
- The logged PESQ for every saved epoch is in [`pesq_curve.csv`](pesq_curve.csv). It rises from 2.69 at epoch 20 to 2.89 at epoch 500, dipping in the middle while the learning rate is high.

## Metrics

These follow the repo's [metrics docs](../../docs/docs/metrics.md):
- DNSMOS P.808 and P.835 (SIG/BAK/OVL).
- SCOREQ (natural domain, NMR mode with the clean reference; lower is better).
- SI-SDR without mean subtraction.
- PESQ P.862.2 wideband.
- STOI and ESTOI.
- WER from Whisper large-v3-turbo, using the upstream `transcript_testset.txt`.

## Training setup

| | |
|---|---|
| Model | FastEnhancer-T (`fastenhancer.default`, channels 24, 2 RNNFormer blocks) |
| Data | VoiceBank-DEMAND, 28-speaker train set (11,572 pairs), resampled 48 → 16 kHz with `scripts.resample` (soxr_hq) |
| Batch | 64 × 2 s random crops (32,000 samples), fp16 mixed precision |
| Loss | 0.3 mag MSE + 0.2 complex MSE + 0.3 consistency + 0.2 wav L1 + 1e-3 PESQ loss |
| Optimizer | AdamP, lr 2e-3, cosine schedule with 500 warmup iterations, 500 epochs (90,500 steps) |
| Hardware | 1× RTX 3090 24 GB (RunPod Community Cloud), torch 2.7.1+cu128 |
| Wall time | 4.9 h of epoch time (35 s/epoch on average), about 2.9 GB of GPU memory |
| Cost | About $1.7 for the whole pod session (7.5 h, including setup, data prep and evaluation) |
| Code | Fork branch `runpod-training`. Includes the `device_id` fix from [aask1357/fastenhancer#20](https://github.com/aask1357/fastenhancer/pull/20) |

## Files

- `model_00500.pth` … `model_00400.pth`: model-only state dicts (`{"model": state_dict, "epoch": N}`) for the five evaluated checkpoints, about 150 KB each. They load with `wrapper.load(path=...)` or `Model(**config["model_kwargs"]).load_state_dict(ckpt["model"])`.
- `config.yaml`: the exact training config.
- `training_log.txt`: per-epoch train and validation losses and logged metrics.

## Reproducing

On a CUDA pod, from a fresh clone of the fork:

```bash
bash scripts/runpod/train_t.sh     # env, data download and resample, upstream test run, training
bash scripts/runpod/eval_top5.sh   # top-5 checkpoint evaluation -> /workspace/results/
```
