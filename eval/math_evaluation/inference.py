#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import ray

from math_eval import AVAILABLE_GPUS, MathEvalWorker, prepare_data
from utils import set_seed


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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_rank_outputs", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


def _init_ray(world_size: int):
    if ray.is_initialized():
        return

    ray_address = os.environ.get("RAY_ADDRESS")
    if ray_address:
        print(f"检测到 RAY_ADDRESS={ray_address}，连接到已有 Ray 集群")
        ray.init(address=ray_address, ignore_reinit_error=True)
        return

    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print("已连接到已有 Ray 集群 (address=auto)")
    except Exception:
        print(f"未发现可连接的 Ray 集群，启动本地 Ray (num_gpus={world_size})")
        ray.init(num_gpus=world_size, ignore_reinit_error=True)


def run_inference_with_ray(args, data_name, examples):
    world_size = len(AVAILABLE_GPUS)
    print(f"使用 {world_size} 个GPU: {AVAILABLE_GPUS}")
    _init_ray(world_size)

    print("创建 workers...")
    workers = [
        MathEvalWorker.remote(
            model_path=args.model_name_or_path,
            gpu_id=gpu_id,
            use_vllm=args.use_vllm,
            apply_chat_template=args.apply_chat_template,
            use_safetensors=args.use_safetensors,
            gpu_memory_utilization=0.95,
        )
        for gpu_id in AVAILABLE_GPUS
    ]

    chunks = np.array_split(examples, world_size)
    chunks = [chunk.tolist() for chunk in chunks if len(chunk) > 0]

    print(f"分配样本到 {len(chunks)} 个 workers:")
    for i, chunk in enumerate(chunks):
        print(f"  Worker {i} (GPU {AVAILABLE_GPUS[i]}): {len(chunk)} 样本")

    args_dict = vars(args)
    print("提交任务...")
    futures = [
        worker.process_batch.remote(
            samples=chunk,
            data_name=data_name,
            args_dict=args_dict,
            rank=i,
        )
        for i, (worker, chunk) in enumerate(zip(workers, chunks))
    ]

    print("等待结果...")
    all_results = ray.get(futures)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_rank_outputs:
        for i, result in enumerate(all_results):
            rank_file = os.path.join(args.output_dir, f"{data_name}_results_rank_{i}.json")
            with open(rank_file, "w", encoding="utf8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    all_samples = []
    total_time = 0.0
    for result in all_results:
        all_samples.extend(result["samples"])
        total_time += result["time_use_in_second"]

    all_samples.sort(key=lambda x: x["idx"])
    final_result_file = os.path.join(args.output_dir, f"{data_name}_final_results.json")
    with open(final_result_file, "w", encoding="utf8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    inference_meta = {
        "num_samples": len(all_samples),
        "time_use_in_second": total_time,
        "time_use_in_minute": f"{int(total_time // 60)}:{int(total_time % 60):02d}",
    }
    meta_file = os.path.join(args.output_dir, f"{data_name}_inference.json")
    with open(meta_file, "w", encoding="utf8") as f:
        json.dump(inference_meta, f, ensure_ascii=False, indent=2)

    print(f"{data_name} 推理完成，结果已保存到: {final_result_file}")
    return inference_meta


def setup(args):
    if args.data_name:
        args.data_names = args.data_name

    data_list = [d.strip() for d in args.data_names.split(",") if d.strip()]
    results = []

    try:
        for data_name in data_list:
            print("\n" + "=" * 50)
            print(f"推理数据集: {data_name}")
            print("=" * 50)

            final_result_file = os.path.join(args.output_dir, f"{data_name}_final_results.json")
            if os.path.exists(final_result_file) and not args.overwrite:
                print(f"检测到 {final_result_file} 已存在，跳过。可使用 --overwrite 覆盖。")
                continue

            examples = prepare_data(data_name, args)
            print(f"样本数量: {len(examples)}")
            if len(examples) > 0:
                print(f"示例: {examples[0]}")

            result = run_inference_with_ray(args, data_name, examples)
            results.append((data_name, result))
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("\nRay shutdown完成")

    if results:
        summary = {name: result for name, result in results}
        with open(os.path.join(args.output_dir, "overall_inference.json"), "w", encoding="utf8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
