#!/usr/bin/env bash
set -euo pipefail

PROMPT_TYPE="alpaca"
# qwen25-math-cot
# PROMPT_TYPE="qwen25-math-cot"
N_SAMPLING=16
TEMPERATURE=1
DATA_NAME="math_oai,minerva_math,olympiadbench,aime24,aime25,amc23"
SPLIT="test"
NUM_TEST_SAMPLE=-1
OUTPUT_ROOT="eval/matheval/outputs_alpaca"
CHECKPOINT_DIR=""
MODEL_DIRS=(
    "/volume/demo/xlzhuang/zh/models/Qwen2.5-7B"
    "/volume/demo/xlzhuang/zh/models/Qwen3-8B-Base"
)

usage() {
    cat <<EOF
Usage:
  $0 [--checkpoint_dir DIR] [--model_dir DIR ...] [--output_root DIR]
     [--prompt_type TYPE] [--n_sampling N] [--temperature T]
     [--data_name NAMES] [--split SPLIT] [--num_test_sample N]

Examples:
  $0 --checkpoint_dir /path/to/ckpts --prompt_type plain
  $0 --model_dir /path/to/model --output_root /path/to/out
Note: --checkpoint_dir will auto-evaluate all model subfolders (or the dir itself if it is a model).
EOF
}

is_model_dir() {
    local d="$1"
    [[ -f "$d/config.json" ]] || \
    [[ -f "$d/model.safetensors" ]] || \
    [[ -f "$d/pytorch_model.bin" ]] || \
    [[ -f "$d/adapter_model.bin" ]] || \
    [[ -f "$d/adapter_config.json" ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --model_dir)
            MODEL_DIRS+=("$2")
            shift 2
            ;;
        --output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --n_sampling)
            N_SAMPLING="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --data_name)
            DATA_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_test_sample)
            NUM_TEST_SAMPLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -n "${CHECKPOINT_DIR}" ]]; then
    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "Checkpoint dir not found: ${CHECKPOINT_DIR}" >&2
        exit 1
    fi
    if is_model_dir "${CHECKPOINT_DIR}"; then
        MODEL_DIRS+=("${CHECKPOINT_DIR}")
    else
        while IFS= read -r d; do
            if is_model_dir "$d"; then
                MODEL_DIRS+=("$d")
            fi
        done < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d | sort)
    fi
fi

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "No model directories found. Use --checkpoint_dir or --model_dir." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

for MODEL_PATH in "${MODEL_DIRS[@]}"; do
    NAME=$(basename "${MODEL_PATH}")
    OUTPUT_DIR="${OUTPUT_ROOT}/${NAME}"

    echo "=============================================="
    echo "Evaluating MODEL: ${MODEL_PATH}"
    echo "Saving to OUTPUT: ${OUTPUT_DIR}"
    echo "PROMPT_TYPE: ${PROMPT_TYPE}"
    echo "DATA_NAME: ${DATA_NAME}"
    echo "=============================================="

    bash sh/eval.sh \
        "${PROMPT_TYPE}" \
        "${MODEL_PATH}" \
        "${OUTPUT_DIR}" \
        "${N_SAMPLING}" \
        "${TEMPERATURE}" \
        "${DATA_NAME}" \
        "${SPLIT}" \
        "${NUM_TEST_SAMPLE}"
done
