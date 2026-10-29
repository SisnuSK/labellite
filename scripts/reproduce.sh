#!/usr/bin/env bash
# Reproduce every table and figure in results/ from scratch, in pipeline order.
# Needs a CUDA GPU (extract, finetune, shift, explain; k-center active learning) and about
# 21 GB of disk (on our machine: data/raw 17 GB, feature cache 3.5 GB). Downloads, features and
# runs that already exist are reused.
#
#   bash scripts/reproduce.sh            # uses the CUDA extra
#   EXTRA=cpu bash scripts/reproduce.sh  # CPU torch (very slow for the GPU steps)
set -euo pipefail

EXTRA="${EXTRA:-gpu}"
ll() { uv run --extra "$EXTRA" labellite "$@"; }

for ds in eurosat pcam; do
  ll data download --config "configs/datasets/$ds.yaml"
  ll data splits --config "configs/datasets/$ds.yaml"
  ll data stats --config "configs/datasets/$ds.yaml"
done

for model in resnet50 dinov2 clip phikon; do
  for ds in eurosat pcam; do
    ll extract --config "configs/extract/${model}_${ds}.yaml"
  done
done

ll probe --config configs/main.yaml
for ds in eurosat pcam; do
  ll finetune --config "configs/finetune/resnet50_$ds.yaml"
done
ll analyze
ll conformal --config configs/main.yaml
ll shift --config configs/shift.yaml
ll activelearn --config configs/al.yaml
ll explain --config configs/main.yaml
ll headline
