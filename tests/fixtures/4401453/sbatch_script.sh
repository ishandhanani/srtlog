#!/bin/bash
#SBATCH --job-name=agg-kv-dynamo
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1


#SBATCH --account=coreai_tritoninference_triton3
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/%j/logs/sweep_%j.log
#SBATCH --partition=batch


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Orchestrator runs on HOST (needs srun access), spawns containerized workers
# Config: /lustre/fsw/coreai_tritoninference_triton3/idhanani/srt-slurm/recipies/qwen3-32b/agg-kv-dynamo.yaml
# Generated: 20251229_153732

set -e

# Setup directories
WORK_DIR="${PWD}"
OUTPUT_DIR="${WORK_DIR}/outputs/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Redirect stderr to stdout (SLURM captures both via --output)
exec 2>&1

echo "=========================================="
echo "🚀 srtctl Sweep Orchestrator"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: /lustre/fsw/coreai_tritoninference_triton3/idhanani/srt-slurm/recipies/qwen3-32b/agg-kv-dynamo.yaml"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "Container: /lustre/fsw/coreai_tritoninference_triton3/idhanani/lmsysorg+sglang+v0.5.6.post2.sqsh"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Copy config to output directory
cp "/lustre/fsw/coreai_tritoninference_triton3/idhanani/srt-slurm/recipies/qwen3-32b/agg-kv-dynamo.yaml" "${OUTPUT_DIR}/config.yaml"

# Get head node
HEAD_NODE=$(scontrol show hostnames ${SLURM_NODELIST} | head -n1)
echo "Head node: ${HEAD_NODE}"

# Install srtctl using container's pip (host may not have pip)
SRTCTL_SOURCE="/lustre/fsw/coreai_tritoninference_triton3/idhanani/srt-slurm"
SRTCTL_INSTALL_DIR="${OUTPUT_DIR}/.srtctl_install"
mkdir -p "${SRTCTL_INSTALL_DIR}"
export SRTCTL_SOURCE_DIR="${SRTCTL_SOURCE}"

echo ""
echo "Installing srtctl dependencies via container..."
srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" \
    --container-image="/lustre/fsw/coreai_tritoninference_triton3/idhanani/lmsysorg+sglang+v0.5.6.post2.sqsh" \
    --container-mounts="${SRTCTL_SOURCE}:/srtctl-src,${SRTCTL_INSTALL_DIR}:/srtctl-install" \
    pip install --quiet --target=/srtctl-install /srtctl-src

export PYTHONPATH="${SRTCTL_INSTALL_DIR}:${PYTHONPATH}"
echo "Installed to ${SRTCTL_INSTALL_DIR}"


# Custom setup script override from CLI
export SRTCTL_SETUP_SCRIPT="install-sglang-main.sh"


echo "Running orchestrator on host (with srun access)..."
echo ""

# Run orchestrator on the HOST (not in container) so it has access to srun
# The orchestrator will spawn containerized workers
python3 -u -m srtctl.cli.do_sweep "/lustre/fsw/coreai_tritoninference_triton3/idhanani/srt-slurm/recipies/qwen3-32b/agg-kv-dynamo.yaml" 2>&1 | tee "${LOG_DIR}/orchestrator_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sweep completed successfully"
else
    echo "✗ Sweep failed (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}