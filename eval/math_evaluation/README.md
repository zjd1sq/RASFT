### Requirements
You can install the required packages with the following command:
```bash
pip install -r requirements.txt
```

If you want to run inference/evaluation separately:
```bash
# inference dependencies
pip install -r requirement-inference.txt

# evaluation-only dependencies
pip install -r requirement-eval.txt
```

### Evaluation
You can evaluate Qwen2.5/Qwen2-Math-Instruct series model with the following command:
```bash
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# Qwen2-Math-Instruct Series
PROMPT_TYPE="qwen-boxed"
# Qwen2-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2-Math-72B-Instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

### Split Inference And Evaluation
The original all-in-one flow (`sh/eval.sh` -> `math_eval.py`) remains unchanged.

You can run split workflow with new scripts:
```bash
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR="eval/matheval/outputs/qwen25-1p5b"
DATA_NAME="math_oai,minerva_math,olympiadbench,aime24,amc23"

# 1) inference only
bash sh/inference.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR 16 1 $DATA_NAME

# 2) evaluation only (from *_final_results.json)
bash sh/eval_only.sh $PROMPT_TYPE $OUTPUT_DIR $DATA_NAME
```



## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
# matheval
