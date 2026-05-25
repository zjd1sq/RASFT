# RASFT: Rollout-Adaptive Supervised Fine-Tuning for Reasoning


Conventional supervised fine-tuning and many SFT variants still optimize the model mainly toward expert demonstrations. Although expert data is useful for teaching new reasoning patterns, excessive imitation of fixed offline demonstrations may distort the model's original knowledge distribution, reduce generation diversity, and weaken generalization.

We view this issue as an offline learning problem and propose **RASFT**, namely **Rollout-Adaptive Supervised Fine-Tuning for Reasoning**.

For each selected training problem, RASFT performs sampling-based rollouts with the current policy model. The generated rollouts are automatically verified using task-specific correctness rules. Correct on-policy rollouts are then added to the training group of the corresponding problem, reducing the rigidity caused by purely offline expert imitation.

RASFT also adaptively adjusts the learning strength of expert demonstrations according to the model's rollout correctness on each problem. For difficult problems with low rollout correctness, RASFT increases the expert-data learning weight, encouraging the model to quickly acquire new knowledge and reasoning skills. For easier problems with high rollout correctness, RASFT decreases the expert-data learning weight, avoiding unnecessary rigid imitation and better preserving the model's original capabilities.

In addition, RASFT introduces an **inverse importance sampling ratio**. The original model is used as the reference model, while the trainable model is used as the policy model:

$$
\left(
\frac{\pi_{\mathrm{ref}}(y_{i,j}\mid x_i)}
{\pi_{\theta}(y_{i,j}\mid x_i)}
\right)
$$

This ratio helps preserve useful knowledge from the original model and mitigates excessive deviation during offline learning.

Comprehensive experiments demonstrate the effectiveness of RASFT on both mathematical reasoning and code reasoning tasks across five backbone models, including Qwen and Llama.

## Installation

```bash
git clone https://github.com/zjd1sq/RASFT.git
cd RASFT

conda create -n rasft python=3.10
conda activate rasft

pip install -r requirements.txt
```

## Data Preparation

Download the mathematical reasoning datasets:

```bash
python download_math_data.py
```

Download the code reasoning datasets:

```bash
python download_code_data.py
```

## Training with SFT Variants

The script `train_v2.py` contains implementations of several SFT and SFT-related variants, including:

- SFT
- SFT + KL
- DFT
- ASFT
- ProFit

Example command for DFT training:

```bash
python train_v2.py \
  --model_name_or_path Qwen/Qwen2.5-Math-7B \
  --mode dft \
  --global_batch_size 256 \
  --model_max_length 2048 \
  --data_path data/numina_cot_10k.jsonl \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --output_dir ./output/dft/Qwen2.5-Math-7B
```

## Training with RASFT

The script `train_RASFT.py` contains the implementation of the proposed RASFT algorithm.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_RASFT.py \
  --mode continuous_train \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
  --data_path data/numina_cot_10k.jsonl \
  --output_dir output/train_rasft_qwen25_math_15b \
  --total_train_steps 1000 \
  --global_batch_size 256 \
  --per_device_train_batch_size 2 \
  --model_max_length 2048 \
  --warmup_steps 4 \
  --learning_rate 5e-5 \
  --gradient_checkpointing \
  --expert_reward_max 3.0 \
  --expert_reward_min 1.0 \
  --pos_reward 1.0 \
  --neg_reward 0.0 \
  --max_negative_keep 3 \
  --adv_alpha 0.4 \
  --rollout_adv_coef 0.25 \
  --rollout_adv_floor 0.02 \
  --rollout_adv_cap 0.15 \
  --raw_chunk_size 128 \
  --replay_buffer_size 2048 \
  --calibration_sample_size 1000 \
  --gap_batch_size 2 \
  --easy_floor_q 0.10 \
  --start_low_q 0.20 \
  --start_high_q 0.45 \
  --end_low_q 0.50 \
  --end_high_q 0.75 \
  --hard_probe_ceiling_q 0.90 \
  --refresh_interval_steps 32 \
  --final_refresh \
  --main_rollout_ratio 0.35 \
  --hard_probe_ratio 0.03 \
  --rollout_gpus 2,3 \
  --tensor_parallel_size 2 \
  --rollout_num 3 \
  --rollout_total_len 2048 \
  --rollout_max_tokens 1024 \
  --rollout_temperature 0.9 \
  --rollout_top_p 0.95 \
  --rollout_gpu_memory_utilization 0.85 \
  --repetition_penalty 1.1
```


## Acknowledgements

Our implementation is inspired by two recent works on supervised fine-tuning and post-training: 1.On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification and 2.Anchored Supervised Fine-Tuning. We sincerely thank the authors for their valuable contributions to this research direction.
