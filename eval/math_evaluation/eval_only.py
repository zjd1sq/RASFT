#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np

from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_name", default=None, type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="plain", type=str)
    parser.add_argument("--max_num_samples", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    return args


def evaluate_one_dataset(data_name, args):
    final_result_file = os.path.join(args.output_dir, f"{data_name}_final_results.json")
    if not os.path.exists(final_result_file):
        print(f"未找到推理结果文件，跳过: {final_result_file}")
        return None

    metrics_file = os.path.join(args.output_dir, f"{data_name}_metrics.json")
    if os.path.exists(metrics_file) and not args.overwrite:
        print(f"检测到 {metrics_file} 已存在，跳过。可使用 --overwrite 覆盖。")
        with open(metrics_file, "r", encoding="utf8") as f:
            return json.load(f)

    with open(final_result_file, "r", encoding="utf8") as f:
        samples = json.load(f)

    _, result_json = evaluate(
        data_name=data_name,
        prompt_type=args.prompt_type,
        samples=samples,
        max_num_samples=args.max_num_samples,
        execute=True,
    )

    inference_meta_file = os.path.join(args.output_dir, f"{data_name}_inference.json")
    if os.path.exists(inference_meta_file):
        with open(inference_meta_file, "r", encoding="utf8") as f:
            inference_meta = json.load(f)
        result_json["time_use_in_second"] = inference_meta.get("time_use_in_second")
        result_json["time_use_in_minute"] = inference_meta.get("time_use_in_minute")

    with open(metrics_file, "w", encoding="utf8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)

    print(f"{data_name} 评测完成，结果已保存到: {metrics_file}")
    return result_json


def write_overall_metrics(results, output_dir):
    if not results:
        return

    names = [name for name, _ in results]
    metrics = [metric for _, metric in results]
    avg_acc = sum([r.get("acc", 0) for r in metrics]) / len(metrics)

    names.append("avg")
    metrics.append({"acc": avg_acc})

    print("\n" + "=" * 50)
    print("所有结果:")
    print("=" * 50)
    for name, result in zip(names, metrics):
        print(f"{name}: {result.get('acc', 0):.4f}")

    overall = {}
    for name, result in zip(names, metrics):
        overall[name] = {
            "acc": result.get("acc", 0),
            "mean_acc": result.get("mean_acc", None),
            "all_acc": result.get("all_acc", None),
            "num_samples": result.get("num_samples", None),
        }

    dataset_results = metrics[:-1]
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

    overall_file = os.path.join(output_dir, "overall_metrics.json")
    with open(overall_file, "w", encoding="utf8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    if args.data_name:
        args.data_names = args.data_name

    data_list = [d.strip() for d in args.data_names.split(",") if d.strip()]
    results = []
    for data_name in data_list:
        print("\n" + "=" * 50)
        print(f"评测数据集: {data_name}")
        print("=" * 50)
        result = evaluate_one_dataset(data_name, args)
        if result is not None:
            results.append((data_name, result))

    write_overall_metrics(results, args.output_dir)


if __name__ == "__main__":
    main()
