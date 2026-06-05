set -euo pipefail

PROMPT_TYPE=${1:-plain}
MODEL_NAME_OR_PATH=${2:-}
OUTPUT_DIR=${3:-}
n_sampling=${4:-1}
temperature=${5:-0}

DATA_NAME=${6:-${DATA_NAME:-"math_oai,minerva_math,olympiadbench,aime24,amc23"}}
SPLIT=${7:-${SPLIT:-"test"}}
NUM_TEST_SAMPLE=${8:-${NUM_TEST_SAMPLE:-"-1"}}

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
    echo "Usage: $0 PROMPT_TYPE MODEL_PATH [OUTPUT_DIR] [N_SAMPLING] [TEMPERATURE] [DATA_NAME] [SPLIT] [NUM_TEST_SAMPLE]" >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="eval/matheval/outputs/$(basename "${MODEL_NAME_OR_PATH}")"
fi

TOKENIZERS_PARALLELISM=false \
python3 -u inference.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_names ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${temperature} \
    --n_sampling ${n_sampling} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm
