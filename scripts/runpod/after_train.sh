#!/usr/bin/env bash
# Wait for training to exit, then run the final top-5 evaluation.
# Progress is written to /workspace/status for polling from outside the pod.
name=vbd/16khz/fastenhancer_t
repo=/workspace/fastenhancer
echo TRAINING > /workspace/status
while pgrep -f "train.py -n $name" >/dev/null; do sleep 60; done
if [[ -f $repo/logs/$name/00500.pth ]]; then
  echo EVALUATING > /workspace/status
  if bash "$repo/scripts/runpod/eval_top5.sh" > /workspace/eval.log 2>&1; then
    echo EVAL_DONE > /workspace/status
  else
    echo EVAL_FAILED > /workspace/status
  fi
else
  echo TRAINING_EXITED_EARLY > /workspace/status
fi
