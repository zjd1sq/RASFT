import os
import sys
import json
import math
import shutil
import argparse
import subprocess
import re
import gc
import random
from collections import deque
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from sympy import sympify, simplify
from sympy.core.sympify import SympifyError

try:
    from sympy.parsing.latex import parse_latex
    HAS_LATEX_PARSER = True
except Exception:
    HAS_LATEX_PARSER = False


IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)



# ============================================================
# Basic utils
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(items: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unwrap_model(model):
    return model


def token_len_of_text(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_runtime():
    return {
        "is_distributed": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
    }


def is_main_process(runtime_info: Dict[str, Any]) -> bool:
    return True


def barrier(runtime_info: Dict[str, Any]):
    return


def broadcast_object(obj, runtime_info: Dict[str, Any]):
    return obj


def shard_list(full_list: List[Any], rank: int, world_size: int) -> List[Any]:
    return full_list


def get_model_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        devices = []
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, str) and dev.startswith("cuda"):
                devices.append(dev)
            elif isinstance(dev, int):
                devices.append(f"cuda:{dev}")
        if devices:
            devices = sorted(set(devices), key=lambda x: int(x.split(":")[1]))
            return torch.device(devices[0])
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ref_forward_device(ref_model) -> torch.device:
    return get_model_input_device(ref_model)


# ============================================================
# Answer extraction / correctness
# ============================================================

def extract_boxed_content(text: str) -> Optional[str]:
    if text is None:
        return None

    key = r"\boxed{"
    last_pos = text.rfind(key)
    if last_pos == -1:
        return None

    i = last_pos + len(key)
    depth = 1
    start = i

    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1

    return None


def strip_math_wrappers(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\:", "")
    s = s.replace("\n", " ")
    s = s.strip(" .,:;!?\t")
    return s.strip()


def latex_frac_to_plain(s: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = pattern.sub(r"((\1)/(\2))", s)
    return s


def latex_sqrt_to_plain(s: str) -> str:
    return re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)


def latex_pow_to_plain(s: str) -> str:
    return s.replace("^", "**")


def normalize_math_text(s: str) -> str:
    s = strip_math_wrappers(s)
    s = latex_frac_to_plain(s)
    s = latex_sqrt_to_plain(s)
    s = latex_pow_to_plain(s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace(" ", "")
    return s


def extract_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    text = text.strip()

    boxed = extract_boxed_content(text)
    if boxed:
        return strip_math_wrappers(boxed)

    patterns = [
        r"final answer is\s*[:：]?\s*(.+)$",
        r"answer is\s*[:：]?\s*(.+)$",
        r"therefore.*?is\s*[:：]?\s*(.+)$",
        r"thus.*?is\s*[:：]?\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            cand = strip_math_wrappers(m.group(1))
            if cand:
                return cand

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    if nums:
        return nums[-1]

    return None


def sympy_parse_expr(expr: str):
    expr_clean = strip_math_wrappers(expr)

    if expr_clean == "":
        raise SympifyError("empty expr")

    if HAS_LATEX_PARSER and ("\\" in expr_clean):
        try:
            return parse_latex(expr_clean)
        except Exception:
            pass

    expr_norm = normalize_math_text(expr_clean)
    return sympify(expr_norm)


def is_correct(gt: Optional[str], pred: Optional[str], tol: float = 1e-8) -> bool:
    if gt is None or pred is None:
        return False

    gt_s = strip_math_wrappers(gt)
    pred_s = strip_math_wrappers(pred)

    if gt_s == pred_s:
        return True

    try:
        if abs(float(gt_s) - float(pred_s)) <= tol:
            return True
    except Exception:
        pass

    try:
        gt_expr = sympy_parse_expr(gt_s)
        pred_expr = sympy_parse_expr(pred_s)
        diff = simplify(gt_expr - pred_expr)
        if diff == 0:
            return True
        try:
            if abs(float(diff)) <= tol:
                return True
        except Exception:
            pass
    except Exception:
        pass

    return False


# ============================================================
# Text quality / format / shortcut
# ============================================================

def normalize_for_similarity(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\t\r]", " ", text)
    return text


def word_ngrams(text: str, n: int = 3) -> set:
    toks = normalize_for_similarity(text).split()
    if len(toks) < n:
        return set()
    return set(tuple(toks[i:i+n]) for i in range(len(toks) - n + 1))


def ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    ga = word_ngrams(a, n=n)
    gb = word_ngrams(b, n=n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / max(union, 1)


def dedupe_samples_by_ngram(samples: List[str], sim_threshold: float = 0.82) -> List[str]:
    kept = []
    for s in samples:
        duplicated = False
        for t in kept:
            if ngram_jaccard(s, t, n=3) >= sim_threshold:
                duplicated = True
                break
        if not duplicated:
            kept.append(s)
    return kept


def has_obvious_loop(text: str) -> bool:
    if not text:
        return True

    norm = normalize_for_similarity(text)

    short_lines = [x.strip() for x in re.split(r"[.\n]", norm) if x.strip()]
    freq = {}
    for ln in short_lines:
        if len(ln) < 12:
            continue
        freq[ln] = freq.get(ln, 0) + 1
        if freq[ln] >= 3:
            return True

    words = norm.split()
    if len(words) >= 30:
        seen = {}
        for i in range(0, len(words) - 7):
            key = " ".join(words[i:i+8])
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                return True

    bad_markers = [
        "let me correct",
        "correcting the calculation",
        "i need to correct",
        "rechecking",
        "let's correct",
        "wait,",
        "however, this is incorrect",
        "the previous reasoning was wrong",
    ]
    hit = sum(1 for x in bad_markers if x in norm)
    if hit >= 2:
        return True

    return False


def looks_gibberish_or_collapsed(text: str) -> bool:
    if text is None:
        return True

    s = text.strip()
    if len(s) < 20:
        return True

    if has_obvious_loop(s):
        return True

    non_word = sum(1 for c in s if not c.isalnum() and not c.isspace())
    if len(s) > 0 and (non_word / len(s)) > 0.42:
        return True

    return False


SHORTCUT_PATTERNS = [
    r"according to the problem[, ]+the correct answer is",
    r"based on the given information[, ]+we know that the correct answer is",
    r"the correct answer is\s*[:：]?\s*\\boxed",
    r"therefore[, ]+the correct answer is",
    r"thus[, ]+the correct answer is",
]


def is_shortcut_sample(text: str) -> bool:
    if text is None:
        return True

    s = text.strip()
    sl = s.lower()

    for pat in SHORTCUT_PATTERNS:
        if re.search(pat, sl):
            return True

    boxed = extract_boxed_content(s)
    if boxed is not None:
        boxed_clean = strip_math_wrappers(boxed)
        if re.fullmatch(r"[A-Da-d①②③④⑤⑥⑦⑧⑨0-9,， ]{1,12}", boxed_clean):
            if len(s) < 180:
                return True

    if boxed is not None and len(s) < 120:
        return True

    return False


def has_valid_format(text: str) -> bool:
    if text is None:
        return False

    s = text.strip()
    if len(s) < 80:
        return False

    if looks_gibberish_or_collapsed(s):
        return False

    boxed = extract_boxed_content(s)
    if boxed is None:
        return False

    sl = s.lower()
    markers = [
        "=",
        "therefore",
        "thus",
        "hence",
        "so ",
        "let ",
        "we have",
        "we get",
        "substitute",
        "solve",
    ]
    if not any((m in sl) or (m in s) for m in markers):
        return False

    return True


def compute_format_and_answer_reward(
    gt_answer: str,
    pred_answer: Optional[str],
    text: str,
    format_reward: float = 0.5,
    answer_reward: float = 1.0,
) -> Tuple[bool, float, bool]:
    format_ok = has_valid_format(text)
    if not format_ok:
        return False, 0.0, False

    correct = is_correct(gt_answer, pred_answer)
    reward = format_reward + (answer_reward if correct else 0.0)
    return True, reward, correct


# ============================================================
# Tokenization / cache
# ============================================================

def truncate_response_to_fit(tokenizer, instruction: str, response: str, max_len: int) -> Optional[str]:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    prompt_len = token_len_of_text(tokenizer, prompt)
    budget = max_len - prompt_len - 1
    if budget <= 16:
        return None

    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if len(resp_ids) <= budget:
        return response

    clipped_ids = resp_ids[:budget]
    clipped = tokenizer.decode(clipped_ids, skip_special_tokens=True)
    return clipped.strip() if len(clipped.strip()) >= 20 else None


def build_tokenized_item(
    tokenizer,
    instruction: str,
    response: str,
    max_len: int,
    advantage: float = 1.0,
    source: str = "expert",
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    clipped_response = truncate_response_to_fit(tokenizer, instruction, response, max_len)
    if clipped_response is None:
        return None

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    full_text = prompt + clipped_response + tokenizer.eos_token

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = full_enc.input_ids[0]
    labels = input_ids.clone()

    prompt_len = min(prompt_enc.input_ids.shape[1], input_ids.shape[0])
    labels[:prompt_len] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)

    if (labels != IGNORE_INDEX).sum().item() < 4:
        return None

    out = {
        "instruction": instruction,
        "response": clipped_response,
        "input_ids": input_ids.cpu(),
        "labels": labels.cpu(),
        "attention_mask": attention_mask.cpu(),
        "advantage": float(advantage),
        "source": source,
    }
    if meta:
        out.update(meta)
    return out


def pretokenize_raw_data(raw_data: List[Dict], tokenizer, max_len: int) -> List[Dict]:
    cached = []
    dropped = 0
    for raw_idx, x in enumerate(raw_data):
        item = build_tokenized_item(
            tokenizer=tokenizer,
            instruction=x["instruction"],
            response=x["response"],
            max_len=max_len,
            advantage=1.0,
            source="expert",
            meta={"raw_idx": raw_idx},
        )
        if item is None:
            dropped += 1
            continue
        cached.append(item)
    print(f"[Pretokenize] kept={len(cached)}, dropped={dropped}")
    return cached


# ============================================================
# Batch assemble
# ============================================================

def collate_tokenized_items(items: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = [x["input_ids"] for x in items]
    labels = [x["labels"] for x in items]
    attention_mask = [x["attention_mask"] for x in items]
    advantages = [torch.tensor(float(x.get("advantage", 1.0)), dtype=torch.float32) for x in items]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    advantages = torch.stack(advantages, dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "advantage": advantages,
    }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.items = deque()

    def add_many(self, items: List[Dict]):
        for x in items:
            self.items.append(x)
            while len(self.items) > self.max_size:
                self.items.popleft()

    def sample_global(self, n: int) -> List[Dict]:
        if len(self.items) == 0 or n <= 0:
            return []
        n = min(n, len(self.items))
        idx = np.random.choice(len(self.items), size=n, replace=False)
        buffer_list = list(self.items)
        return [buffer_list[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def dump_jsonl(self, path: str):
        save_jsonl(list(self.items), path)


# ============================================================
# Difficulty = current model seq loss on expert sequence
# ============================================================

@torch.no_grad()
def compute_seq_loss_batch(
    model,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = out.logits[:, :-1].contiguous()
    shift_labels = batch["labels"][:, 1:].to(logits.device).contiguous()
    valid_mask = shift_labels.ne(IGNORE_INDEX)

    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, T)

    valid_len = valid_mask.sum(-1).clamp(min=1)
    seq_loss = (ce * valid_mask).sum(-1) / valid_len
    return seq_loss.detach().cpu()


def compute_seq_loss_for_items(
    model,
    items: List[Dict],
    pad_token_id: int,
    model_input_device: torch.device,
    batch_size: int,
) -> List[float]:
    vals = []
    model.eval()
    for start in range(0, len(items), batch_size):
        sub = items[start:start + batch_size]
        batch = collate_tokenized_items(sub, pad_token_id)
        batch = move_batch_to_device(batch, model_input_device)
        seq_loss = compute_seq_loss_batch(model, batch)
        vals.extend(seq_loss.tolist())
        if ((start // batch_size) + 1) % 8 == 0:
            cleanup_cuda()
    cleanup_cuda()
    return vals


# ============================================================
# Training loss / step with ref model constraint
# ============================================================
def compute_weighted_loss(model, ref_model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits
    device = logits.device

    labels = batch["labels"].to(device)
    advantages = batch["advantage"].to(device)

    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(IGNORE_INDEX)

    B, T, V = shift_logits.shape

    token_ce = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(B, T)

    p_new = torch.exp(-token_ce)

    with torch.no_grad():
        ref_device = get_ref_forward_device(ref_model)
        ref_outputs = ref_model(
            input_ids=batch["input_ids"].to(ref_device),
            attention_mask=batch["attention_mask"].to(ref_device),
            use_cache=False,
        )
        ref_logits = ref_outputs.logits[:, :-1].contiguous().to(device)

        ref_ce = F.cross_entropy(
            ref_logits.view(-1, ref_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        ).view(B, T)

        logp_old = -ref_ce
        logp_new = -token_ce

        seq_len = valid_mask.sum(-1).clamp(min=1)
        seq_logp_new = (logp_new * valid_mask).sum(-1) / seq_len
        seq_logp_old = (logp_old * valid_mask).sum(-1) / seq_len

        log_ratio = torch.clamp(seq_logp_old - seq_logp_new, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        ratio_clip = torch.clamp(ratio, 0.4, 1.2)

        weight = (
            ratio_clip.unsqueeze(-1)
            * advantages.unsqueeze(-1).to(p_new.dtype)
            * p_new
        ).detach()

    loss = token_ce * weight
    loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
    return loss


# ============================================================
# Rollout filtering / reward / soft advantage
# ============================================================
def classify_and_filter_rollout_samples_unified(
    instruction: str,
    gt_answer: str,
    samples: List[str],             # 删除了 expert_response
    tokenizer,
    max_len: int,
    max_negative_keep: int = 1,
    expert_reward_max: float = 3.0,
    expert_reward_min: float = 1.5,
    pos_reward: float = 1.0,        # 名字统一改为 pos_reward
    neg_reward: float = 0.0,
) -> Optional[Dict]:
    if gt_answer is None:
        return None

    samples = dedupe_samples_by_ngram(samples, sim_threshold=0.82)
    k_total_samples = len(samples) 

    processed = []
    for s in samples:
        if s is None: continue
        s = s.strip()
        if len(s) < 20 or looks_gibberish_or_collapsed(s) or is_shortcut_sample(s): continue

        clipped = truncate_response_to_fit(tokenizer, instruction, s, max_len)
        if clipped is None or is_shortcut_sample(clipped) or not has_valid_format(clipped): continue

        pred = extract_answer(clipped)
        correct = is_correct(gt_answer, pred)
        processed.append({
            "response": clipped,
            "correct": bool(correct),
        })

    if len(processed) == 0:
        return None

    positives = [x for x in processed if x["correct"]]
    negatives = [x for x in processed if not x["correct"]]

    kept_neg = []
    if len(negatives) > 0 and max_negative_keep > 0:
        for neg in negatives:
            sims = [ngram_jaccard(neg["response"], pos["response"], n=3) for pos in positives]
            max_sim = max(sims) if sims else 0.0
            if max_sim >= 0.88: continue
            kept_neg.append((neg, max_sim))
            
        kept_neg = sorted(kept_neg, key=lambda x: abs(x[1] - 0.45))
        kept_neg = [{"response": x[0]["response"], "correct": False} for x in kept_neg[:max_negative_keep]]

    k_pos = len(positives)
    difficulty_ratio = 1.0 - (k_pos / max(k_total_samples, 1))
    expert_reward = expert_reward_min + difficulty_ratio * (expert_reward_max - expert_reward_min)

    # 修改这里：使用 pos_reward
    for pos in positives:
        pos["reward"] = pos_reward

    for neg in kept_neg:
        neg["reward"] = neg_reward

    group_rewards = [expert_reward]
    group_rewards.extend([float(x["reward"]) for x in positives])
    group_rewards.extend([float(x["reward"]) for x in kept_neg])

    if len(group_rewards) < 2:
        return None

    arr = np.array(group_rewards, dtype=np.float32)
    if np.allclose(arr, arr[0]):
        return None

    return {
        "expert_reward": expert_reward,
        "train_rollouts": positives,
        "stat_negatives": kept_neg,
        "group_rewards": group_rewards,
    }




def compute_rollout_soft_advantages(
    rewards: List[float],
    adv_alpha: float = 0.6,
) -> Optional[List[float]]:
    if len(rewards) == 0:
        return None

    arr = np.array(rewards, dtype=np.float32)
    if np.allclose(arr, arr[0]):
        return None

    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return None

    z = (arr - mean) / (std + 1e-6)
    soft_adv = np.tanh(adv_alpha * z) # 纯粹的 GRPO/PPO baseline 近似
    return soft_adv.tolist()


# ============================================================
# Rollout worker
# ============================================================
def rollout_worker_main(args):
    from vllm import LLM, SamplingParams

    prompts_data = load_jsonl(args.prompt_file)

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        trust_remote_code=False,
        max_model_len=max(args.model_max_length, args.rollout_total_len),
        enforce_eager=True,
    )

    buckets = {}
    for meta in prompts_data:
        prompt_len = int(meta["prompt_token_len"])
        max_new_tokens = max(
            32,
            min(
                args.rollout_max_tokens,
                args.rollout_total_len - prompt_len - 8,
                args.model_max_length - prompt_len - 8,
            )
        )
        if max_new_tokens < 32:
            continue
        buckets.setdefault(max_new_tokens, []).append(meta)

    results = []

    try:
        for max_new_tokens, bucket_items in buckets.items():
            prompts = [x["prompt"] for x in bucket_items]

            sampling_params = SamplingParams(
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                n=args.rollout_num,
                max_tokens=max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )

            outputs = llm.generate(prompts, sampling_params)

            for meta, out in zip(bucket_items, outputs):
                samples = [o.text for o in out.outputs]
                results.append({
                    "raw_idx": meta["raw_idx"],
                    "instruction": meta["instruction"],
                    "samples": samples,
                })

        results.sort(key=lambda x: x["raw_idx"])
        save_jsonl(results, args.sample_file)

        print(
            f"[Rollout Worker] saved {len(results)} results to {args.sample_file}",
            flush=True,
        )

    except Exception as e:
        print(f"[Rollout Worker] failed: {repr(e)}", flush=True)
        raise

    finally:
        # 避免 vLLM / CUDA / multiprocessing 资源析构卡死
        try:
            del llm
        except Exception:
            pass

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        # 关键：worker 已经完成任务，强制退出子进程
        os._exit(0)



def save_rollout_snapshot(model, tokenizer, path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    ensure_dir(path)
    unwrap_model(model).save_pretrained(path)
    tokenizer.save_pretrained(path)


def run_rollout_subprocess(model_dir: str, prompts_data: List[Dict], args, refresh_id: int):
    ensure_dir(args.output_dir)

    prompt_file = os.path.join(args.output_dir, f"refresh_{refresh_id}_prompts.jsonl")
    sample_file = os.path.join(args.output_dir, f"refresh_{refresh_id}_samples.jsonl")

    save_jsonl(prompts_data, prompt_file)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.rollout_gpus

    tp_size = len([x for x in args.rollout_gpus.split(",") if x.strip() != ""])

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "rollout_worker",
        "--model_name_or_path", model_dir,
        "--prompt_file", prompt_file,
        "--sample_file", sample_file,
        "--rollout_num", str(args.rollout_num),
        "--rollout_temperature", str(args.rollout_temperature),
        "--rollout_top_p", str(args.rollout_top_p),
        "--rollout_max_tokens", str(args.rollout_max_tokens),
        "--rollout_total_len", str(args.rollout_total_len),
        "--model_max_length", str(args.model_max_length),
        "--tensor_parallel_size", str(tp_size),
        "--rollout_gpu_memory_utilization", str(args.rollout_gpu_memory_utilization),
        "--repetition_penalty", str(args.repetition_penalty),
    ]

    print(
        f"[Rollout Subprocess] start refresh={refresh_id}, "
        f"num_prompts={len(prompts_data)}, "
        f"rollout_num={args.rollout_num}, "
        f"gpus={args.rollout_gpus}",
        flush=True,
    )

    completed = subprocess.run(cmd, check=True, env=env)

    print(
        f"[Rollout Subprocess] finished refresh={refresh_id}, "
        f"returncode={completed.returncode}",
        flush=True,
    )

    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Rollout sample file not found: {sample_file}")

    sample_results = load_jsonl(sample_file)

    print(
        f"[Rollout Subprocess] loaded {len(sample_results)} rollout results from {sample_file}",
        flush=True,
    )

    return sample_results



# ============================================================
# Threshold calibration + sliding window
# ============================================================

def calibrate_loss_thresholds(
    model,
    raw_cache: List[Dict],
    tokenizer,
    model_input_device: torch.device,
    args,
):
    if len(raw_cache) == 0:
        raise ValueError("raw_cache is empty")

    if args.calibration_sample_size > 0 and len(raw_cache) > args.calibration_sample_size:
        idx = np.random.choice(len(raw_cache), size=args.calibration_sample_size, replace=False)
        sample_items = [raw_cache[i] for i in idx.tolist()]
    else:
        sample_items = raw_cache

    losses = compute_seq_loss_for_items(
        model=model,
        items=sample_items,
        pad_token_id=tokenizer.pad_token_id,
        model_input_device=model_input_device,
        batch_size=args.gap_batch_size,
    )
    arr = np.array(losses, dtype=np.float32)

    thresholds = {
        "easy_floor": float(np.quantile(arr, args.easy_floor_q)),
        "start_low": float(np.quantile(arr, args.start_low_q)),
        "start_high": float(np.quantile(arr, args.start_high_q)),
        "end_low": float(np.quantile(arr, args.end_low_q)),
        "end_high": float(np.quantile(arr, args.end_high_q)),
        "hard_probe_ceiling": float(np.quantile(arr, args.hard_probe_ceiling_q)),
    }
    return thresholds


def current_window_thresholds(thresholds: Dict[str, float], step: int, total_steps: int):
    alpha = min(max(step / max(total_steps, 1), 0.0), 1.0)
    low = (1 - alpha) * thresholds["start_low"] + alpha * thresholds["end_low"]
    high = (1 - alpha) * thresholds["start_high"] + alpha * thresholds["end_high"]
    return {
        "easy_floor": thresholds["easy_floor"],
        "window_low": float(low),
        "window_high": float(high),
        "hard_probe_ceiling": thresholds["hard_probe_ceiling"],
        "alpha": alpha,
    }


# ============================================================
# Continuous main loop
# ============================================================

def load_tokenizer_and_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_data = load_jsonl(args.data_path)
    if args.data_fraction < 1.0:
        keep_n = max(1, int(len(raw_data) * args.data_fraction))
        raw_data = raw_data[:keep_n]
        print(f"[Quick Run] using first {keep_n} samples ({args.data_fraction:.0%} of full data)")

    return tokenizer, raw_data


def build_refresh_prompt_items(raw_cache: List[Dict], tokenizer, selected_raw_idx: List[int], args):
    prompt_items = []
    for raw_idx in selected_raw_idx:
        instruction = raw_cache[raw_idx]["instruction"]
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        prompt_token_len = token_len_of_text(tokenizer, prompt)
        if prompt_token_len >= args.rollout_total_len - 32:
            continue
        prompt_items.append({
            "raw_idx": raw_idx,
            "instruction": instruction,
            "prompt": prompt,
            "prompt_token_len": prompt_token_len,
        })
    return prompt_items


def route_chunk_by_difficulty(
    chunk_indices: List[int],
    chunk_losses: List[float],
    window: Dict[str, float],
):
    easy_direct = []
    mid_pending = []
    hard_pool_direct = []
    hard_never = []

    for raw_idx, loss in zip(chunk_indices, chunk_losses):
        if loss <= window["easy_floor"]:
            easy_direct.append(raw_idx)
        elif window["window_low"] <= loss <= window["window_high"]:
            mid_pending.append(raw_idx)
        elif loss > window["window_high"] and loss <= window["hard_probe_ceiling"]:
            hard_pool_direct.append(raw_idx)
        else:
            hard_never.append(raw_idx)

    return easy_direct, mid_pending, hard_pool_direct, hard_never


def enqueue_expert_item(expert_queue: deque, raw_cache: List[Dict], raw_idx: int, advantage: float):
    base = raw_cache[raw_idx]
    item = dict(base)
    item["advantage"] = float(advantage)
    item["source"] = "expert"
    expert_queue.append(item)

            
def apply_refresh(
    policy_model,
    tokenizer,
    raw_cache: List[Dict],
    pending_mid: List[int],
    hard_pool: Dict[int, Dict[str, Any]],
    expert_queue: deque,
    replay_buffer: ReplayBuffer,
    thresholds: Dict[str, float],
    refresh_id: int,
    current_step: int,
    args,
    runtime_info,
):
    window = current_window_thresholds(thresholds, current_step, args.total_train_steps)

    # weak-positive rollout controls
    rollout_adv_coef = getattr(args, "rollout_adv_coef", 0.25)
    rollout_adv_floor = getattr(args, "rollout_adv_floor", 0.02)
    rollout_adv_cap = getattr(args, "rollout_adv_cap", 0.15)

    selected_main = []
    if len(pending_mid) > 0:
        k = int(max(1, math.ceil(len(pending_mid) * args.main_rollout_ratio)))
        k = min(k, len(pending_mid))
        selected_main = random.sample(pending_mid, k)

    selected_probe = []
    hard_keys = list(hard_pool.keys())
    if len(hard_keys) > 0 and args.hard_probe_ratio > 0:
        k = int(len(hard_keys) * args.hard_probe_ratio)
        if k == 0:
            k = 1
        k = min(k, len(hard_keys))
        selected_probe = random.sample(hard_keys, k)

    decision = {
        "selected_main": sorted(selected_main),
        "selected_probe": sorted(selected_probe),
        "window": window,
    }
    decision = broadcast_object(decision, runtime_info)
    selected_main = decision["selected_main"]
    selected_probe = decision["selected_probe"]
    window = decision["window"]

    selected_main_set = set(selected_main)

    # 没被 rollout 的 mid 样本直接走 expert SFT
    for raw_idx in pending_mid:
        if raw_idx not in selected_main_set:
            enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)

    probe_rollout_candidates = []
    if len(selected_probe) > 0:
        probe_items = [raw_cache[i] for i in selected_probe]
        probe_losses = compute_seq_loss_for_items(
            model=policy_model,
            items=probe_items,
            pad_token_id=tokenizer.pad_token_id,
            model_input_device=get_model_input_device(policy_model),
            batch_size=args.gap_batch_size,
        )

        for raw_idx, loss in zip(selected_probe, probe_losses):
            if loss < window["window_low"]:
                if raw_idx in hard_pool:
                    del hard_pool[raw_idx]
                enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
            elif loss <= window["hard_probe_ceiling"]:
                probe_rollout_candidates.append(raw_idx)
            else:
                if raw_idx in hard_pool:
                    hard_pool[raw_idx]["times_probed"] += 1
                    hard_pool[raw_idx]["last_probe_step"] = current_step

    probe_rollout_candidates = broadcast_object(probe_rollout_candidates, runtime_info)

    selected_all = sorted(list(set(selected_main + probe_rollout_candidates)))

    queued_rollouts = 0

    if len(selected_all) > 0:
        snapshot_dir = os.path.join(args.output_dir, f"refresh_{refresh_id}_snapshot")
        prompt_file = os.path.join(args.output_dir, f"refresh_{refresh_id}_prompts.jsonl")
        sample_file = os.path.join(args.output_dir, f"refresh_{refresh_id}_samples.jsonl")

        save_rollout_snapshot(policy_model, tokenizer, snapshot_dir)
        barrier(runtime_info)

        try:
            prompt_items = build_refresh_prompt_items(raw_cache, tokenizer, selected_all, args)
            sample_results = run_rollout_subprocess(
                model_dir=snapshot_dir,
                prompts_data=prompt_items,
                args=args,
                refresh_id=refresh_id,
            )

            for item in sample_results:
                raw_idx = item["raw_idx"]
                instruction = raw_cache[raw_idx]["instruction"]
                expert_response = raw_cache[raw_idx]["response"]
                gt_answer = extract_answer(expert_response)

                filtered = classify_and_filter_rollout_samples_unified(
                    instruction=instruction,
                    gt_answer=gt_answer,
                    samples=item["samples"],
                    tokenizer=tokenizer,
                    max_len=args.model_max_length,
                    max_negative_keep=args.max_negative_keep,
                    expert_reward_max=args.expert_reward_max,
                    expert_reward_min=args.expert_reward_min,
                    pos_reward=getattr(args, "pos_reward", 1.0),
                    neg_reward=args.neg_reward,
                )

                # 包括：
                # 1. 全部 rollout 被过滤
                # 2. 全对且 expert_reward_min == pos_reward 导致 group_rewards 零方差
                # 3. 其他无法构造有效 advantage 的情况
                #
                # 统一 fallback：只训 expert
                if filtered is None:
                    enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
                    continue

                train_rollouts = filtered["train_rollouts"]
                k_pos = len(train_rollouts)

                group_advantages = compute_rollout_soft_advantages(
                    rewards=filtered["group_rewards"],
                    adv_alpha=args.adv_alpha,
                )

                # 理论上 classify 里已经挡了一部分零方差；
                # 这里保留防御逻辑。
                if group_advantages is None:
                    if k_pos > 0:
                        # 认为模型已经会了：不训练 rollout，避免污染；
                        # hard_pool 中也删掉，避免反复 probe。
                        if raw_idx in hard_pool:
                            del hard_pool[raw_idx]
                        enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
                        continue
                    else:
                        # 完全失败：只训 expert
                        enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
                        continue

                # 一个正确 rollout 都没有：
                # 错误 rollout 不入队训练，只训 expert。
                if k_pos == 0:
                    enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
                    continue

                expert_group_adv = float(group_advantages[0])
                pos_group_advs = group_advantages[1:1 + k_pos]

                # 单题更新阻尼器：一题里正确 rollout 越多，每条样本权重越小。
                scale = 1.0 / float(1 + k_pos)

                # expert 是高质量主锚点，保留原来的动态 advantage。
                expert_adv = expert_group_adv * scale

                enqueue_expert_item(
                    expert_queue,
                    raw_cache,
                    raw_idx,
                    advantage=expert_adv,
                )

                # correct rollout：弱正向学习
                # 目标：
                # 1. 正确 rollout 永远不允许负梯度；
                # 2. 给一个很小正下限，保留多样性泛化信号；
                # 3. 封顶，避免低质量但答案正确的 rollout 权重过大；
                # 4. 再乘 rollout_adv_coef，让它显著弱于 expert。
                for rollout_item, adv in zip(train_rollouts, pos_group_advs):
                    rollout_adv = float(adv) * scale

                    # 不允许 correct rollout 负梯度
                    rollout_adv = max(rollout_adv_floor, rollout_adv)

                    # 防止 correct rollout 权重太大
                    rollout_adv = min(rollout_adv_cap, rollout_adv)

                    # correct rollout 总体降权
                    rollout_adv = rollout_adv * rollout_adv_coef

                    tok_item = build_tokenized_item(
                        tokenizer=tokenizer,
                        instruction=instruction,
                        response=rollout_item["response"],
                        max_len=args.model_max_length,
                        advantage=rollout_adv,
                        source="rollout",
                        meta={
                            "raw_idx": raw_idx,
                            "refresh_id": refresh_id,
                        },
                    )
                    if tok_item is not None:
                        expert_queue.append(tok_item)
                        queued_rollouts += 1

                if raw_idx in hard_pool:
                    del hard_pool[raw_idx]

        finally:
            barrier(runtime_info)

            if is_main_process(runtime_info):
                cleanup_targets = [snapshot_dir, prompt_file, sample_file]
                for path in cleanup_targets:
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.isfile(path):
                            os.remove(path)
                    except Exception as e:
                        print(f"[Refresh {refresh_id}] cleanup failed for {path}: {e}")

            barrier(runtime_info)

    print(
        f"[Refresh {refresh_id}] "
        f"pending_mid={len(pending_mid)} "
        f"selected_main={len(selected_main)} "
        f"selected_probe={len(selected_probe)} "
        f"probe_rollout={len(probe_rollout_candidates)} "
        f"queued_rollouts={queued_rollouts} "
        f"queue={len(expert_queue)} "
        f"hard_pool={len(hard_pool)} "
        f"window=[{window['window_low']:.4f}, {window['window_high']:.4f}] "
        f"alpha={window['alpha']:.4f} "
        f"rollout_adv_coef={rollout_adv_coef:.4f} "
        f"rollout_adv_floor={rollout_adv_floor:.4f} "
        f"rollout_adv_cap={rollout_adv_cap:.4f}"
    )

    return hard_pool, []


def continuous_train(args):
    runtime_info = init_runtime()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    tokenizer, raw_data = load_tokenizer_and_data(args)
    raw_cache = pretokenize_raw_data(raw_data, tokenizer, args.model_max_length)
    if len(raw_cache) == 0:
        raise ValueError("No valid tokenized raw samples.")

    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    print("[Load] policy model with device_map='auto' ...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    policy_model.config.use_cache = False
    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        try:
            policy_model.enable_input_require_grads()
        except Exception:
            pass

    policy_input_device = get_model_input_device(policy_model)
    print(f"[Policy input device] {policy_input_device}")
    if hasattr(policy_model, "hf_device_map"):
        print(f"[Policy device map] {policy_model.hf_device_map}")

    print("[Load] ref model with device_map='auto' ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    ref_input_device = get_ref_forward_device(ref_model)
    print(f"[Ref input device] {ref_input_device}")
    if hasattr(ref_model, "hf_device_map"):
        print(f"[Ref device map] {ref_model.hf_device_map}")

    thresholds = calibrate_loss_thresholds(
        model=policy_model,
        raw_cache=raw_cache,
        tokenizer=tokenizer,
        model_input_device=policy_input_device,
        args=args,
    )
    print("[Calibration thresholds]", thresholds)

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_train_steps,
    )

    replay_buffer = ReplayBuffer(max_size=args.replay_buffer_size)

    expert_queue = deque()
    pending_mid: List[int] = []
    hard_pool: Dict[int, Dict[str, Any]] = {}

    raw_cursor = 0
    optimizer_step = 0
    micro_step = 0
    refresh_id = 0
    running_loss = 0.0

    micro_batch = args.per_device_train_batch_size

    def flush_hard_pool_to_queue():
        if len(hard_pool) == 0:
            return
        hard_keys = sorted(list(hard_pool.keys()))
        print(f"[HardPool Flush] move {len(hard_keys)} hard samples to train queue.")
        for raw_idx in hard_keys:
            enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
        hard_pool.clear()

    while optimizer_step < args.total_train_steps and (
        raw_cursor < len(raw_cache)
        or len(expert_queue) > 0
        or len(pending_mid) > 0
        or len(hard_pool) > 0
    ):
        if raw_cursor < len(raw_cache):
            chunk_end = min(len(raw_cache), raw_cursor + args.raw_chunk_size)
            chunk_items = raw_cache[raw_cursor:chunk_end]
            chunk_indices = [x["raw_idx"] for x in chunk_items]

            chunk_losses = compute_seq_loss_for_items(
                model=policy_model,
                items=chunk_items,
                pad_token_id=tokenizer.pad_token_id,
                model_input_device=policy_input_device,
                batch_size=args.gap_batch_size,
            )
            window = current_window_thresholds(thresholds, optimizer_step, args.total_train_steps)
            easy_direct, mid_pending_chunk, hard_pool_direct, hard_never = route_chunk_by_difficulty(
                chunk_indices=chunk_indices,
                chunk_losses=chunk_losses,
                window=window,
            )

            print(
                f"[Route raw {raw_cursor}:{chunk_end}] "
                f"easy={len(easy_direct)} mid={len(mid_pending_chunk)} "
                f"hard_pool={len(hard_pool_direct)} hard_never={len(hard_never)} "
                f"window=[{window['window_low']:.4f}, {window['window_high']:.4f}]"
            )

            for raw_idx in easy_direct:
                enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)

            for raw_idx in mid_pending_chunk:
                pending_mid.append(raw_idx)

            for raw_idx in hard_pool_direct:
                enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)
                if raw_idx not in hard_pool:
                    hard_pool[raw_idx] = {
                        "last_seen_step": optimizer_step,
                        "times_probed": 0,
                        "last_probe_step": -1,
                    }

            for raw_idx in hard_never:
                enqueue_expert_item(expert_queue, raw_cache, raw_idx, advantage=1.0)

            raw_cursor = chunk_end

        if (
            optimizer_step > 0
            and optimizer_step % args.refresh_interval_steps == 0
            and len(pending_mid) > 0
        ):

            refresh_id += 1
            hard_pool, pending_mid = apply_refresh(
                policy_model=policy_model,
                tokenizer=tokenizer,
                raw_cache=raw_cache,
                pending_mid=pending_mid,
                hard_pool=hard_pool,
                expert_queue=expert_queue,
                replay_buffer=replay_buffer,
                thresholds=thresholds,
                refresh_id=refresh_id,
                current_step=optimizer_step,
                args=args,
                runtime_info=runtime_info,
            )

        if len(expert_queue) == 0 and len(pending_mid) > 0:
            refresh_id += 1
            hard_pool, pending_mid = apply_refresh(
                policy_model=policy_model,
                tokenizer=tokenizer,
                raw_cache=raw_cache,
                pending_mid=pending_mid,
                hard_pool=hard_pool,
                expert_queue=expert_queue,
                replay_buffer=replay_buffer,
                thresholds=thresholds,
                refresh_id=refresh_id,
                current_step=optimizer_step,
                args=args,
                runtime_info=runtime_info,
            )

        if (
            len(expert_queue) == 0
            and raw_cursor >= len(raw_cache)
            and len(pending_mid) == 0
            and len(hard_pool) > 0
        ):
            flush_hard_pool_to_queue()

        if len(expert_queue) == 0 and raw_cursor >= len(raw_cache) and len(pending_mid) == 0 and len(hard_pool) == 0:
            break

        batch_items = []
        take_n = min(micro_batch, len(expert_queue))
        for _ in range(take_n):
            batch_items.append(expert_queue.popleft())

        if len(batch_items) == 0:
            continue

        batch = collate_tokenized_items(batch_items, tokenizer.pad_token_id)
        batch = move_batch_to_device(batch, policy_input_device)

        loss = compute_weighted_loss(policy_model, ref_model, batch)
        loss = loss / args.grad_accum_steps
        loss.backward()

        micro_step += 1
        running_loss += float(loss.detach().item())

        if micro_step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % args.logging_steps == 0:
                avg_loss = running_loss / max(args.logging_steps, 1)
                running_loss = 0.0
                print(
                    f"[Train] step={optimizer_step} "
                    f"loss={avg_loss:.6f} "
                    f"queue={len(expert_queue)} "
                    f"pending_mid={len(pending_mid)} "
                    f"hard_pool={len(hard_pool)}"
                )

    if args.final_refresh and len(pending_mid) > 0:
        refresh_id += 1
        hard_pool, pending_mid = apply_refresh(
            policy_model=policy_model,
            tokenizer=tokenizer,
            raw_cache=raw_cache,
            pending_mid=pending_mid,
            hard_pool=hard_pool,
            expert_queue=expert_queue,
            replay_buffer=replay_buffer,
            thresholds=thresholds,
            refresh_id=refresh_id,
            current_step=optimizer_step,
            args=args,
            runtime_info=runtime_info,
        )

    if len(hard_pool) > 0:
        flush_hard_pool_to_queue()

    while len(expert_queue) > 0 and optimizer_step < args.total_train_steps:
        batch_items = []
        take_n = min(micro_batch, len(expert_queue))
        for _ in range(take_n):
            batch_items.append(expert_queue.popleft())

        if len(batch_items) == 0:
            break

        batch = collate_tokenized_items(batch_items, tokenizer.pad_token_id)
        batch = move_batch_to_device(batch, policy_input_device)

        loss = compute_weighted_loss(policy_model, ref_model, batch)
        loss = loss / args.grad_accum_steps
        loss.backward()

        micro_step += 1
        running_loss += float(loss.detach().item())

        if micro_step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % args.logging_steps == 0:
                avg_loss = running_loss / max(args.logging_steps, 1)
                running_loss = 0.0
                print(
                    f"[Final Drain] step={optimizer_step} "
                    f"loss={avg_loss:.6f} "
                    f"queue={len(expert_queue)}"
                )

    final_dir = os.path.join(args.output_dir, "final_model")
    save_rollout_snapshot(policy_model, tokenizer, final_dir)
    replay_buffer.dump_jsonl(os.path.join(args.output_dir, "rollout_replay_buffer.jsonl"))
    with open(os.path.join(args.output_dir, "hard_gap_pool.json"), "w", encoding="utf-8") as f:
        json.dump(hard_pool, f)
    with open(os.path.join(args.output_dir, "calibrated_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print(f"[Done] final model saved to {final_dir}")



# ============================================================
# Args
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="continuous_train",
        choices=["continuous_train", "rollout_worker"]
    )

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output/train_oarpo_v2_auto")

    parser.add_argument("--total_train_steps", type=int, default=1000)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--data_fraction", type=float, default=1.0)

    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--raw_chunk_size", type=int, default=64)
    parser.add_argument("--replay_buffer_size", type=int, default=2048)

    parser.add_argument("--calibration_sample_size", type=int, default=1000)
    parser.add_argument("--gap_batch_size", type=int, default=8)

    parser.add_argument("--easy_floor_q", type=float, default=0.10)
    parser.add_argument("--start_low_q", type=float, default=0.20)
    parser.add_argument("--start_high_q", type=float, default=0.45)
    parser.add_argument("--end_low_q", type=float, default=0.50)
    parser.add_argument("--end_high_q", type=float, default=0.75)
    parser.add_argument("--hard_probe_ceiling_q", type=float, default=0.95)

    parser.add_argument("--refresh_interval_steps", type=int, default=32)
    parser.add_argument("--final_refresh", action="store_true")
    parser.add_argument("--main_rollout_ratio", type=float, default=0.35)
    parser.add_argument("--hard_probe_ratio", type=float, default=0.03)

    parser.add_argument("--rollout_num", type=int, default=3)
    parser.add_argument("--rollout_temperature", type=float, default=0.9)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--rollout_max_tokens", type=int, default=1024)
    parser.add_argument("--rollout_total_len", type=int, default=2048)
    parser.add_argument("--rollout_gpus", type=str, default="2,3")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--rollout_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    
    # ==========================================
    # OARPO v2 核心参数：动态难度锚点 + 固定质量奖励
    # ==========================================
    parser.add_argument("--expert_reward_max", type=float, default=3.0)
    parser.add_argument("--expert_reward_min", type=float, default=1.0)
    parser.add_argument("--pos_reward", type=float, default=1.0)
    parser.add_argument("--neg_reward", type=float, default=0.0)
    parser.add_argument("--max_negative_keep", type=int, default=3)
    parser.add_argument("--adv_alpha", type=float, default=0.4)

    # correct rollout weak-positive controls
    parser.add_argument("--rollout_adv_coef", type=float, default=0.25)
    parser.add_argument("--rollout_adv_floor", type=float, default=0.02)
    parser.add_argument("--rollout_adv_cap", type=float, default=0.15)


    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--sample_file", type=str, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode != "rollout_worker" and args.data_path is None:
        raise ValueError("--data_path is required")

    if args.mode == "continuous_train":
        denom = args.per_device_train_batch_size
        if args.global_batch_size % denom != 0:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} not divisible by "
                f"per_device_train_batch_size={denom}"
            )
        args.grad_accum_steps = args.global_batch_size // denom
    else:
        args.grad_accum_steps = 1

    if args.mode == "rollout_worker":
        rollout_worker_main(args)
    elif args.mode == "continuous_train":
        continuous_train(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
