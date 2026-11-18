#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == path-singles || ${1:-} == path-doubles ]]; then
    comparison=$1
    if (( $# != 3 )); then
        echo "Usage: bash scripts/compare.sh {path-singles|path-doubles} GPU_IDS DATA_PREFIX" >&2
        exit 2
    fi
    devices=$2 data=$3
    run_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    repo=$(cd "$run_dir/.." && pwd)
    export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
    mkdir -p "$repo/runs"
    out=$(mktemp -d "$repo/runs/$comparison.XXXXXX")
    echo "Three-op training logs: $out"
    export MEGATRON_LAYER_TIMING=14 MEGATRON_MODEL_TIMING=1
    export MEGATRON_TRAIN_ITERS=${MEGATRON_TRAIN_ITERS:-201}
    export MEGATRON_TIMEOUT_SECONDS=${MEGATRON_TIMEOUT_SECONDS:-600}
    warmup=${MEGATRON_TIMING_WARMUP:-20}
    if [[ ! $MEGATRON_TRAIN_ITERS =~ ^[1-9][0-9]*$ || ! $warmup =~ ^[0-9]+$ ]] \
        || (( warmup >= MEGATRON_TRAIN_ITERS )); then
        echo "Expected training iterations greater than timing warmup" >&2
        exit 2
    fi
    case "$comparison" in
        path-singles) order=(original paths_ag paths_rs paths_ar paths_ag_rs_ar) ;;
        path-doubles) order=(original paths_ag_rs paths_ag_ar paths_rs_ar paths_ag_rs_ar) ;;
    esac
    for ((i=${#order[@]}-1; i>=0; i--)); do
        order+=("${order[i]}")
    done

    check_pair_processes() {
        local root=$1 report=$2 gpu_pids observed_at
        gpu_pids=$(nvidia-smi --id="$devices" --query-compute-apps=pid --format=csv,noheader,nounits) || return
        observed_at=$(date +%s) || return
        ps -e -o pid=,ppid= | awk -v root="$root" -v running="$gpu_pids" -v at="$observed_at" '
            { parent[$1] = $2 }
            END {
                n = split(running, pids)
                for (i = 1; i <= n; i++) {
                    pid = pids[i]
                    if (pid !~ /^[0-9]+$/ || !(pid in parent)) continue
                    while (pid > 1 && pid != root && pid in parent) pid = parent[pid]
                    if (pid != root) print "foreign_gpu_process=" pids[i] " observed_at=" at
                }
            }
        ' >> "$report"
    }

    slot=0
    for arm in "${order[@]}"; do
        slot=$((slot + 1))
        report="$out/$slot-$arm.gpu.log"
        MEGATRON_RUN_DIR="$out/$slot-$arm" THREEOPS_CONFIG="$arm" \
            bash "$run_dir/train.sh" "$devices" 4096 "$data" &
        job=$!
        while kill -0 "$job" 2>/dev/null; do
            if ! check_pair_processes "$job" "$report"; then
                echo "gpu_process_query_failed" >> "$report"
                break
            fi
            sleep 2
        done
        wait "$job"
        if [[ -s $report ]]; then
            python3 -c 'from pathlib import Path; import sys; from tilescale_threeops.analyze import check_timing_overlap; check_timing_overlap(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))' \
                "$report" "$out/$slot-$arm/train.log" "$warmup" 
        fi
    done
    python3 -m tilescale_threeops.analyze "$comparison" "$out" \
        --steps "$MEGATRON_TRAIN_ITERS" --warmup "$warmup"
    exit
fi
echo "Use path-singles or path-doubles" >&2
exit 2
