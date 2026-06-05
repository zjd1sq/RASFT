#!/usr/bin/env python3
import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np

import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions


def _load_available_gpus():
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpus:
        return [gid.strip() for gid in env_gpus.split(",") if gid.strip()]
    return ["0", "1", "2", "3"]


AVAILABLE_GPUS = _load_available_gpus()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_name", default=None, type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="plain", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    return examples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


# ==================== Ray Worker ====================

@ray.remote(num_gpus=1)
class MathEvalWorker:
    """Ray actor for math evaluation on a single GPU"""
    
    def __init__(
        self, 
        model_path: str, 
        gpu_id: str, 
        use_vllm: bool,
        apply_chat_template: bool,
        use_safetensors: bool,
        gpu_memory_utilization: float = 0.95  # ✅ 保留原始值
    ):
        """
        Initialize worker on specific GPU
        
        Args:
            model_path: Path to the model
            gpu_id: GPU ID to use
            use_vllm: Whether to use vLLM
            apply_chat_template: Whether to apply chat template
            use_safetensors: Whether to use safetensors
            gpu_memory_utilization: GPU memory utilization ratio
        """
        # 设置 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[Worker GPU {gpu_id}] Initializing model: {model_path}")
        
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.use_vllm = use_vllm
        self.apply_chat_template = apply_chat_template
        
        # ✅ 保留两种模型加载方式
        if use_vllm:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            self.tokenizer = None
            if apply_chat_template:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
        else:
            # ✅ 保留 HuggingFace 模式
            import torch
            self.llm, self.tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=model_path,
                load_in_half=True,
                use_fast_tokenizer=True,
                use_safetensors=use_safetensors,
            )
            self.llm = self.llm.cuda()
        
        print(f"[Worker GPU {gpu_id}] Model loaded successfully")
    
    def process_batch(
        self,
        samples: list,
        data_name: str,
        args_dict: dict,
        rank: int
    ) -> dict:
        """Process a batch of examples"""
        print(f"[Worker GPU {self.gpu_id}] Processing {len(samples)} samples")
        
        # Reconstruct args
        class Args:
            pass
        args = Args()
        for key, value in args_dict.items():
            setattr(args, key, value)
        
        # Initialize executor
        if "pal" in args.prompt_type:
            executor = PythonExecutor(get_answer_expr="solution()")
        else:
            executor = PythonExecutor(get_answer_from_stdout=True)
        
        # Prepare prompts
        processed_samples = []
        for example in tqdm(samples, desc=f"GPU {self.gpu_id} - Preparing"):
            idx = example["idx"]
            
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans
            full_prompt = construct_prompt(example, data_name, args)
            
            sample = {
                "idx": idx,
                "question": example["question"],
                "gt_cot": gt_cot,
                "gt": gt_ans,
                "prompt": full_prompt,
            }
            
            for key in [
                "level", "type", "unit", "solution_type", "choices", "solution",
                "ques_type", "ans_type", "answer_type", "dataset", "subfield",
                "filed", "theorem", "answer",
            ]:
                if key in example:
                    sample[key] = example[key]
            
            processed_samples.append(sample)
        
        # Prepare input prompts
        input_prompts = [
            sample["prompt"] for sample in processed_samples 
            for _ in range(args.n_sampling)
        ]
        
        if self.apply_chat_template and self.tokenizer:
            input_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in input_prompts
            ]
        
        # Configure stop words
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", 
                      "<|end_of_text|>", "<｜end▁of▁sentence｜>"]
        
        if args.prompt_type in ["cot"]:
            stop_words.append("\n\nQuestion:")
        if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
            stop_words.extend(["\n\n---", "```output"])
        elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
            stop_words.extend(["Instruction", "Response"])
        elif "jiuzhang" in args.prompt_type:
            stop_words.append("\n\n## Question")
        elif "numina" in args.prompt_type:
            stop_words.append("\n### Problem")
        elif "pure" in args.prompt_type:
            stop_words.append("\n\n\n")
        
        if "qwen2" in self.model_path.lower():
            stop_token_ids = [151645, 151643]
        elif "deepseek" in self.model_path.lower():
            stop_token_ids = [100001]
        else:
            stop_token_ids = None
        
        # Generate
        print(f"[Worker GPU {self.gpu_id}] Generating {len(input_prompts)} outputs")
        start_time = time.time()
        
        # ✅ 保留两种生成方式
        if self.use_vllm:
            outputs = self.llm.generate(
                input_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=stop_token_ids,
                ),
            )
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = generate_completions(
                model=self.llm,
                tokenizer=self.tokenizer,
                prompts=input_prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )
        
        time_use = time.time() - start_time
        print(f"[Worker GPU {self.gpu_id}] Generation completed in {time_use:.2f}s")
        
        # Clean outputs
        codes = []
        for i in range(len(input_prompts)):
            output = outputs[i].rstrip()
            code = output
            for stop_word in stop_words:
                if stop_word in code:
                    code = code.split(stop_word)[0].strip()
            codes.append(code)
        
        # Execute
        results = [
            run_execute(executor, code, args.prompt_type, data_name) 
            for code in codes
        ]
        
        # Organize results
        all_samples = []
        for i, sample in enumerate(processed_samples):
            code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
            preds = [item[0] for item in result]
            reports = [item[1] for item in result]
            
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A", "B", "C", "D", "E",
                ]:
                    preds[j] = choice_answer_clean(code[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )
            
            sample.pop("prompt")
            sample.update({"code": code, "pred": preds, "report": reports})
            all_samples.append(sample)
        
        return {
            "samples": all_samples,
            "time_use_in_second": time_use,
            "time_use_in_minute": f"{int(time_use // 60)}:{int(time_use % 60):02d}"
        }


def evaluate_with_ray(args, data_name, examples):
    """Evaluate using Ray-distributed workers"""
    world_size = len(AVAILABLE_GPUS)
    print(f"使用 {world_size} 个GPU: {AVAILABLE_GPUS}")
    
    if not ray.is_initialized():
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            print(f"检测到 RAY_ADDRESS={ray_address}，连接到已有 Ray 集群")
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            try:
                # Prefer attaching to an existing cluster when available.
                ray.init(address="auto", ignore_reinit_error=True)
                print("已连接到已有 Ray 集群 (address=auto)")
            except Exception:
                print(f"未发现可连接的 Ray 集群，启动本地 Ray (num_gpus={world_size})")
                ray.init(num_gpus=world_size, ignore_reinit_error=True)
    
    try:
        # Create workers
        print("创建 workers...")
        workers = [
            MathEvalWorker.remote(
                model_path=args.model_name_or_path,
                gpu_id=gpu_id,
                use_vllm=args.use_vllm,  # ✅ 传递参数
                apply_chat_template=args.apply_chat_template,
                use_safetensors=args.use_safetensors,  # ✅ 传递参数
                gpu_memory_utilization=0.95  # ✅ 保留原始值
            )
            for gpu_id in AVAILABLE_GPUS
        ]
        
        # Split examples
        chunks = np.array_split(examples, world_size)
        chunks = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
        
        print(f"分配样本到 {len(chunks)} 个 workers:")
        for i, chunk in enumerate(chunks):
            print(f"  Worker {i} (GPU {AVAILABLE_GPUS[i]}): {len(chunk)} 样本")
        
        # Convert args to dict
        args_dict = vars(args)
        
        # Submit tasks
        print("提交任务...")
        futures = [
            worker.process_batch.remote(
                samples=chunk,
                data_name=data_name,
                args_dict=args_dict,
                rank=i
            )
            for i, (worker, chunk) in enumerate(zip(workers, chunks))
        ]
        
        # Collect results
        print("等待结果...")
        all_results = ray.get(futures)
        
        # ✅ 保存每个 rank 的中间结果（可选，用于调试）
        os.makedirs(args.output_dir, exist_ok=True)
        for i, result in enumerate(all_results):
            rank_file = os.path.join(
                args.output_dir, 
                f"{data_name}_results_rank_{i}.json"
            )
            with open(rank_file, "w", encoding="utf8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Merge
        all_samples = []
        total_time = 0
        for result in all_results:
            all_samples.extend(result["samples"])
            total_time += result["time_use_in_second"]
        
        # Sort by index
        all_samples.sort(key=lambda x: x["idx"])
        
        print(f"收集到 {len(all_samples)} 个结果")
        
        # Save merged results
        with open(
            os.path.join(args.output_dir, f"{data_name}_final_results.json"), 
            "w", 
            encoding="utf8"
        ) as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        # Evaluate
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )
        
        # Save metrics
        result_json["time_use_in_second"] = total_time
        result_json["time_use_in_minute"] = f"{int(total_time // 60)}:{int(total_time % 60):02d}"
        
        with open(
            os.path.join(args.output_dir, f"{data_name}_metrics.json"), 
            "w"
        ) as f:
            json.dump(result_json, f, indent=4)
        
        # ✅ 删除中间文件（可选）
        # for i in range(world_size):
        #     rank_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{i}.json")
        #     if os.path.exists(rank_file):
        #         os.remove(rank_file)
        
        print(f"{data_name} 评估完成！")
        print(f"准确率: {result_json.get('acc', 0):.4f}")
        
        return result_json
    
    finally:
        pass


def setup(args):
    """Main setup and evaluation loop"""
    
    if args.data_name:
        args.data_names = args.data_name
    data_list = args.data_names.split(",")
    results = []
    
    for data_name in data_list:
        print("\n" + "=" * 50)
        print(f"评估数据集: {data_name}")
        print("=" * 50)
        
        final_result_file = os.path.join(
            args.output_dir, 
            f"{data_name}_final_results.json"
        )
        if os.path.exists(final_result_file):
            print(f"检测到 {final_result_file} 已存在，跳过。")
            metrics_file = os.path.join(args.output_dir, f"{data_name}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    result_json = json.load(f)
                results.append(result_json)
            continue
        
        examples = prepare_data(data_name, args)
        print(f"样本数量: {len(examples)}")
        if len(examples) > 0:
            print(f"示例: {examples[0]}")
        
        result_json = evaluate_with_ray(args, data_name, examples)
        results.append(result_json)
    
    if ray.is_initialized():
        ray.shutdown()
        print("\nRay shutdown完成")
    
    if results:
        data_list.append("avg")
        results.append({
            "acc": sum([r.get("acc", 0) for r in results]) / len(results),
        })
        
        print("\n" + "=" * 50)
        print("所有结果:")
        print("=" * 50)
        for name, result in zip(data_list, results):
            print(f"{name}: {result.get('acc', 0):.4f}")

        # Write overall metrics summary
        overall = {}
        for name, result in zip(data_list, results):
            overall[name] = {
                "acc": result.get("acc", 0),
                "mean_acc": result.get("mean_acc", None),
                "all_acc": result.get("all_acc", None),
                "num_samples": result.get("num_samples", None),
            }

        # Add aggregate metrics across datasets (simple mean, not sample-weighted)
        dataset_results = [r for r in results[:-1]]  # exclude "avg" entry
        if dataset_results:
            acc_vals = [r.get("acc", 0) for r in dataset_results]
            mean_acc_vals = [r.get("mean_acc", 0) for r in dataset_results]
            overall["overall"] = {
                "acc": float(np.mean(acc_vals)),
                "mean_acc": float(np.mean(mean_acc_vals)),
                "all_acc": None,
                "num_samples": int(sum([r.get("num_samples", 0) or 0 for r in dataset_results])),
            }
            all_acc_lists = [r.get("all_acc") for r in dataset_results if r.get("all_acc") is not None]
            if all_acc_lists:
                min_len = min(len(x) for x in all_acc_lists)
                if min_len > 0:
                    trimmed = [x[:min_len] for x in all_acc_lists]
                    overall["overall"]["all_acc"] = list(np.mean(np.array(trimmed), axis=0))

        overall_file = os.path.join(args.output_dir, "overall_metrics.json")
        with open(overall_file, "w") as f:
            json.dump(overall, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
