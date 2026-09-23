"""Evaluate the top-k checkpoints of a finished run with scripts.metrics_ns.

Checkpoints are ranked by the PESQ that training logs to TensorBoard
(`metrics/pesq`, computed every `pesq.interval` epochs on the `pesq` set).
Writes results.csv, results.json, pesq_curve.csv and model-only weights of
the evaluated checkpoints to --out.

Usage (on the pod, from the repo root, inside the venv):
    python -m scripts.runpod.eval_top5 -n vbd/16khz/fastenhancer_t \
        --transcript-dir /workspace/data/voicebank-demand/logfiles/transcript_testset.txt \
        --out /workspace/results/fastenhancer_t_vbd16k
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METRICS = ["p808_mos", "sig", "bak", "ovr", "scoreq", "sisnr", "pesq", "stoi", "estoi", "wer"]


def logged_pesq(base_dir):
    """Latest logged value of metrics/pesq and metrics/stoi per epoch."""
    scores = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "valid", "events.out.tfevents.*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        if "metrics/pesq" not in tags:
            continue
        stoi = {e.step: e.value for e in acc.Scalars("metrics/stoi")} if "metrics/stoi" in tags else {}
        for e in acc.Scalars("metrics/pesq"):
            scores[e.step] = {"pesq": e.value, "stoi": stoi.get(e.step)}
    return scores


def evaluate(name, epoch, transcript_dir, num_threads):
    cmd = [sys.executable, "-m", "scripts.metrics_ns", "-n", name, "-e", str(epoch),
           "-d", "cuda:0", "--num-threads", str(num_threads), "--transcript-dir", transcript_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"metrics_ns failed for epoch {epoch}:\n{proc.stderr[-3000:]}")
    last = [l for l in proc.stdout.replace("\r", "\n").splitlines() if l.startswith(f"{name}: ")][-1]
    values = [float(v) for v in last.split(": ", 1)[1].split(", ")]
    return dict(zip(METRICS, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--jobs", type=int, default=5, help="checkpoints evaluated in parallel")
    a = parser.parse_args()

    base_dir = os.path.join("logs", a.name)
    os.makedirs(a.out, exist_ok=True)
    scores = logged_pesq(base_dir)
    saved = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(base_dir, "*.pth"))}
    candidates = sorted((e for e in scores if e in saved), key=lambda e: -scores[e]["pesq"])
    top = candidates[:a.top_k]
    print("Logged PESQ at saved epochs:", {e: round(scores[e]["pesq"], 4) for e in sorted(candidates)})
    print("Top epochs:", top, flush=True)

    with open(os.path.join(a.out, "pesq_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "logged_pesq", "logged_stoi", "checkpoint_saved"])
        for e in sorted(scores):
            w.writerow([e, f"{scores[e]['pesq']:.4f}",
                        "" if scores[e]["stoi"] is None else f"{scores[e]['stoi']:.4f}", e in saved])

    jobs = max(1, min(a.jobs, len(top)))
    threads = max(1, (os.cpu_count() or 1) // jobs)
    print(f"Evaluating {len(top)} checkpoints, {jobs} in parallel, {threads} threads each...", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(lambda e: evaluate(a.name, e, a.transcript_dir, threads), top))
    rows = []
    for rank, (epoch, metrics) in enumerate(zip(top, results), 1):
        row = {"rank": rank, "epoch": epoch, "logged_pesq": round(scores[epoch]["pesq"], 4)}
        row.update(metrics)
        rows.append(row)
        print(row, flush=True)
        ckpt = torch.load(os.path.join(base_dir, f"{epoch:0>5d}.pth"), map_location="cpu")
        torch.save({"model": ckpt["model"], "epoch": ckpt["epoch"]},
                   os.path.join(a.out, f"model_{epoch:0>5d}.pth"))
    shutil.copy(os.path.join(base_dir, "config.yaml"), os.path.join(a.out, "config.yaml"))

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("Done:", a.out)


if __name__ == "__main__":
    main()
