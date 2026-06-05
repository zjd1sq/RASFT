set -euo pipefail

PROMPT_TYPE=${1:-plain}
OUTPUT_DIR=${2:-}
DATA_NAME=${3:-${DATA_NAME:-"math_oai,minerva_math,olympiadbench,aime24,amc23"}}
MAX_NUM_SAMPLES=${4:-}

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "Usage: $0 PROMPT_TYPE OUTPUT_DIR [DATA_NAME] [MAX_NUM_SAMPLES]" >&2
    exit 1
fi

if [[ -n "${MAX_NUM_SAMPLES}" ]]; then
    python3 -u eval_only.py \
        --prompt_type ${PROMPT_TYPE} \
        --output_dir ${OUTPUT_DIR} \
        --data_names ${DATA_NAME} \
        --max_num_samples ${MAX_NUM_SAMPLES}
else
    python3 -u eval_only.py \
        --prompt_type ${PROMPT_TYPE} \
        --output_dir ${OUTPUT_DIR} \
        --data_names ${DATA_NAME}
fi
