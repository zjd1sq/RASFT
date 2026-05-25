import os
import copy
import logging
import json
import random
import numpy as np
import fire
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType

os.environ["WANDB_MODE"] = "offline"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

PROMPT_DICT = {
    "prompt_no_input": (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "{instruction}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B")

@dataclass
class DataArguments:
    data_path: str = field(default="data/numinamath_10k.jsonl")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    output_dir: str = field(default="./output")
    per_device_train_batch_size: int = field(default=2)
    num_train_epochs: float = field(default=1.0)
    learning_rate: float = field(default=5e-5)


class EnhancedTrainer(Trainer):
    def __init__(self, mode="sft", kl_weight=0.05, clip_min=0.1, clip_max=2.0, alpha=0.1, original_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.kl_weight = kl_weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = alpha
        self.original_model = original_model
        if original_model is not None:
            self.original_model.eval()
        print(f"Training mode: {mode}, kl_weight: {kl_weight}, alpha: {alpha}")

    def get_reference_logits(self, model, inputs):
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                ref_outputs = model(**inputs)
                ref_logits = ref_outputs.logits
        else:
            with torch.no_grad():
                ref_outputs = self.original_model(**inputs)
                ref_logits = ref_outputs.logits
        return ref_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            valid_mask = shift_labels != IGNORE_INDEX

            if valid_mask.sum() == 0:
                loss = torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits, shift_labels)

                if self.mode == "sft":
                    loss = token_losses[valid_mask].sum() / valid_mask.sum()

                elif self.mode == "dft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1) - 1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    weighted_losses = token_losses * weights
                    loss = weighted_losses[valid_mask].sum() / valid_mask.sum()

                elif self.mode == "sft+kl":
                    with torch.no_grad():
                        ref_logits = self.get_reference_logits(model, inputs)
                        ref_logits = ref_logits[..., :-1, :].contiguous()
                        ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:shift_logits.size(0)]
                    kl_div = F.kl_div(
                        F.log_softmax(shift_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    weighted_losses = token_losses + self.kl_weight * kl_div
                    loss = weighted_losses[valid_mask].sum() / valid_mask.sum()

                elif self.mode == "asft":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1) - 1)
                    weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
                    dft_losses = token_losses * weights
                    with torch.no_grad():
                        ref_logits = self.get_reference_logits(model, inputs)
                        ref_logits = ref_logits[..., :-1, :].contiguous()
                        ref_logits = ref_logits.view(-1, ref_logits.size(-1))[:shift_logits.size(0)]
                    kl_div = F.kl_div(
                        F.log_softmax(shift_logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="none",
                    ).sum(dim=-1)
                    weighted_losses = dft_losses + self.kl_weight * kl_div
                    loss = weighted_losses[valid_mask].sum() / valid_mask.sum()

                elif self.mode == "profit":
                    probs = torch.softmax(shift_logits, dim=-1)
                    valid_labels_clamped = torch.clamp(shift_labels, min=0, max=probs.size(-1) - 1)
                    token_probs = probs.gather(1, valid_labels_clamped.unsqueeze(-1)).squeeze(-1).detach()
                    profit_mask = (token_probs > 0.1) & valid_mask
                    if profit_mask.sum() > 0:
                        loss = token_losses[profit_mask].sum() / profit_mask.sum()
                    else:
                        loss = token_losses[valid_mask].sum() / valid_mask.sum()

                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = load_jsonl(data_path)
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [prompt_no_input.format_map(example) for example in list_data_dict]
        targets = [f"{example['response']}{tokenizer.eos_token}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def show_first_example(data_path: str, tokenizer: transformers.PreTrainedTokenizer):
    print("\n" + "=" * 60)
    print("FIRST TRAINING EXAMPLE CHECK")
    print("=" * 60)
    data = load_jsonl(data_path)
    if not data:
        print("No data found")
        return
    example = data[0]
    prompt = PROMPT_DICT["prompt_no_input"].format_map(example)
    response = example.get("response", "")
    full_text = prompt + response + tokenizer.eos_token

    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    src_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    total_len = tokenized.input_ids.shape[1]
    src_len = src_tokenized.input_ids.shape[1]
    print(f"Prompt:\n{prompt}")
    print(f"Response (first 200 chars): {response[:200]}")
    print(f"\nTotal tokens : {total_len}")
    print(f"Prompt tokens: {src_len}")
    print(f"Response tokens (loss region): {total_len - src_len}")
    if total_len >= tokenizer.model_max_length:
        print("⚠️  WARNING: sequence truncated — consider larger model_max_length")
    print("=" * 60 + "\n")


def train(
    model_name_or_path: str = "models/Llama-2-7b",
    data_path: str = "data/train_medmcqa_alpaca_10k.jsonl",
    cache_dir: str = None,
    model_max_length: int = 512,
    per_device_train_batch_size: int = 2,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-5,
    global_batch_size: int = 64,
    mode: str = "sft",  # sft, dft, sft+kl, asft, profit
    kl_weight: float = 0.1,
    alpha: float = 0.1,
    clip_min: float = 0.1,
    clip_max: float = 2.0,
    output_dir: str = None,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    deepspeed_config: Optional[str] = None,
    precision: str = "bf16",
    gradient_checkpointing: bool = False,
    seed: int = 42,
    max_steps: int = -1,
    save_ckpt: bool = False,   # 默认不保存中间 ckpt
    save_steps: int = 500,
    save_total_limit: int = 2,
    **kwargs
):
    """Enhanced training with multiple DFT variants"""

    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    data_args = DataArguments(data_path=data_path)

    set_random_seed(seed)

    print("==== ModelArguments ====")
    print(model_args)
    print("========================")

    print("==== DataArguments ====")
    print(data_args)
    print("=======================")

    if output_dir is None:
        output_dir = f"./output/{mode}/{os.path.basename(model_name_or_path)}"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = max(1, global_batch_size // (per_device_train_batch_size * world_size))

    print("==== Training Parameters ====")
    print(f"model_name_or_path: {model_name_or_path}")
    print(f"data_path: {data_path}")
    print(f"mode: {mode}")
    print(f"global_batch_size: {global_batch_size}")
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"num_train_epochs: {num_train_epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"world_size: {world_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"max_steps: {max_steps}")
    print(f"save_ckpt: {save_ckpt}")
    print(f"save_steps: {save_steps}")
    print("=============================")

    precision = precision.lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}. Use bf16, fp16, or fp32.")
    use_bf16 = precision == "bf16"
    use_fp16 = precision == "fp16"
    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_distributed else 0
    deepspeed_enabled = deepspeed_config is not None

    if is_distributed:
        print(f"Distributed training mode detected: {world_size} GPUs")
    else:
        print("Single GPU training mode")

    if deepspeed_enabled:
        os.environ.setdefault("DEEPSPEED_ZERO_INIT", "0")
        os.environ.setdefault("DEEPSPEED_ZERO3_INIT", "0")

    model_kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": torch_dtype,
    }
    if not is_distributed and not deepspeed_enabled:
        model_kwargs["device_map"] = "auto"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    if use_lora:
        logger.info(f"Using LoRA with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if is_distributed and not deepspeed_enabled:
        model = model.to(f"cuda:{local_rank}")

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model_name_lower = model_args.model_name_or_path.lower()
    llama3_markers = ["llama-3", "llama3", "llama 3", "llama_3", "llama3.", "llama-3."]
    is_llama3_family = any(marker in model_name_lower for marker in llama3_markers)
    if is_llama3_family:
        logger.info(
            "Detected LLaMA 3 family model (%s); skipping manual special token overrides.",
            model_args.model_name_or_path
        )
    elif "llama" in model_name_lower:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })

    show_first_example(data_args.data_path, tokenizer)

    original_model = None
    if ("kl" in mode or mode == "asft") and not use_lora:
        print("Loading original model for KL divergence...")
        original_model_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch_dtype,
        }
        if not is_distributed and not deepspeed_enabled:
            original_model_kwargs["device_map"] = "auto"

        original_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **original_model_kwargs
        )
        if is_distributed:
            original_model = original_model.to(f"cuda:{local_rank}")

        original_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if original_tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=original_tokenizer,
                model=original_model,
            )
        for param in original_model.parameters():
            param.requires_grad = False
        original_model.eval()

    # 默认不保存中间 ckpt
    train_args_dict = dict(
        output_dir=output_dir,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no" if not save_ckpt else "steps",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=seed,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed_config,
        report_to="none",
    )

    if save_ckpt:
        train_args_dict["save_steps"] = save_steps
        train_args_dict["save_total_limit"] = save_total_limit

    train_args_dict.update(kwargs)

    training_args = TrainingArguments(**train_args_dict)

    print("==== Transformers TrainingArguments ====")
    print(training_args)
    print("========================================")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = EnhancedTrainer(
        mode=mode,
        kl_weight=kl_weight,
        alpha=alpha,
        clip_min=clip_min,
        clip_max=clip_max,
        original_model=original_model,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()

    # 只保存 final model
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    fire.Fire(train)

