import argparse
import json
import re
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


# ============================================================
# Common helpers
# ============================================================

QUESTION_CANDIDATES = [
    "question",
    "prompt",
    "instruction",
    "text",
    "problem",
]

RESPONSE_CANDIDATES = [
    "r1_solution",
    "solution",
    "code",
    "response",
    "answer",
    "completion",
]

TEST_CANDIDATES = [
    "test",
    "tests",
    "test_list",
    "unit_tests",
    "test_cases",
]

ENTRY_POINT_CANDIDATES = [
    "entry_point",
    "function_name",
    "fn_name",
    "name",
]


def pick_first(item: Dict[str, Any], candidates: List[str], default=None):
    for k in candidates:
        if k in item and item[k] is not None:
            return item[k]
    return default


def as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, tuple):
        return [str(v) for v in x]
    return [str(x)]


def infer_entry_point_from_tests(test_list: List[str]) -> Optional[str]:
    """
    Infer function name from assert statements, e.g.
    assert add(1, 2) == 3  -> add
    assert set(similar_elements(...)) == ... -> similar_elements
    """
    for t in test_list:
        t = str(t)

        # Direct form: assert func(...)
        m = re.search(r"assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", t)
        if m:
            name = m.group(1)
            if name not in {"set", "list", "tuple", "dict", "len", "sorted", "sum", "all", "any"}:
                return name

        # Wrapped form: assert set(func(...)) == ...
        m = re.search(
            r"assert\s+(?:set|list|tuple|dict|len|sorted|sum|all|any)\s*"
            r"\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            t,
        )
        if m:
            return m.group(1)

    return None


def normalize_test_value(test_value) -> Dict[str, Any]:
    if test_value is None:
        return {}

    if isinstance(test_value, list):
        tests = [str(x) for x in test_value]
        return {
            "tests": tests,
            "test": "\n".join(tests),
            "test_list": tests,
        }

    test_str = str(test_value)
    tests = [line.strip() for line in test_str.splitlines() if line.strip()]
    return {
        "test": test_str,
        "tests": tests,
        "test_list": tests,
    }


def write_jsonl(rows: List[Dict[str, Any]], output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# KodCode train data
# ============================================================

def normalize_kodcode_record(item: Dict[str, Any]) -> Dict[str, Any]:
    instruction = pick_first(item, QUESTION_CANDIDATES, "")
    response = pick_first(item, RESPONSE_CANDIDATES, "")

    out = {
        "instruction": str(instruction) if instruction is not None else "",
        "response": str(response) if response is not None else "",
        "benchmark": "kodcode",
    }

    test_value = pick_first(item, TEST_CANDIDATES, None)
    out.update(normalize_test_value(test_value))

    entry_point = pick_first(item, ENTRY_POINT_CANDIDATES, None)
    if entry_point is None and out.get("test_list"):
        entry_point = infer_entry_point_from_tests(out["test_list"])
    if entry_point is not None:
        out["entry_point"] = str(entry_point)

    # Preserve useful original fields for debugging / later eval
    preserve_keys = [
        "task_id",
        "question",
        "prompt",
        "instruction",
        "text",
        "problem",
        "r1_solution",
        "solution",
        "code",
        "response",
        "answer",
        "difficulty",
        "source",
        "split",
        "category",
        "domain",
    ]

    for k in preserve_keys:
        if k in item and k not in out:
            out[k] = item[k]

    return out


def download_kodcode(args) -> List[Dict[str, Any]]:
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
    )

    if args.streaming:
        dataset = dataset.shuffle(
            seed=args.seed,
            buffer_size=args.shuffle_buffer_size,
        )
        iterator = islice(dataset, args.sample_size)
    else:
        dataset = dataset.shuffle(seed=args.seed)
        if args.sample_size > 0:
            dataset = dataset.select(range(min(args.sample_size, len(dataset))))
        iterator = dataset

    rows = []
    missing_instruction = 0
    missing_response = 0
    with_test = 0
    with_entry_point = 0

    for count, item in enumerate(iterator, start=1):
        row = normalize_kodcode_record(item)

        if not row.get("instruction"):
            missing_instruction += 1
        if not row.get("response"):
            missing_response += 1
        if row.get("test") or row.get("tests") or row.get("test_list"):
            with_test += 1
        if row.get("entry_point"):
            with_entry_point += 1

        rows.append(row)

        if count % args.log_every == 0:
            print(f"Processed {count}/{args.sample_size}")

    print("=" * 80)
    print("[KodCode Download Summary]")
    print("=" * 80)
    print(f"Dataset              : {args.dataset_name}")
    print(f"Split                : {args.split}")
    print(f"Saved rows           : {len(rows)}")
    print(f"Rows with tests      : {with_test}/{len(rows)}")
    print(f"Rows with entry_point: {with_entry_point}/{len(rows)}")
    print(f"Missing instruction  : {missing_instruction}/{len(rows)}")
    print(f"Missing response     : {missing_response}/{len(rows)}")
    if rows:
        print(f"Example output keys  : {list(rows[0].keys())}")

    return rows


# ============================================================
# MBPP eval data
# ============================================================

def normalize_mbpp_record(item: Dict[str, Any], config: str, split: str) -> Dict[str, Any]:
    prompt = item.get("prompt") or item.get("text") or ""
    code = item.get("code") or ""

    test_list = as_list(item.get("test_list"))

    # full config usually has test_setup_code.
    # sanitized config usually has test_imports.
    test_setup_code = item.get("test_setup_code", "")
    test_imports = item.get("test_imports", "")

    if isinstance(test_imports, list):
        test_imports_str = "\n".join(str(x) for x in test_imports)
    else:
        test_imports_str = str(test_imports) if test_imports is not None else ""

    if not test_setup_code:
        test_setup_code = test_imports_str

    challenge_test_list = as_list(item.get("challenge_test_list"))

    entry_point = item.get("entry_point")
    if entry_point is None:
        entry_point = infer_entry_point_from_tests(test_list)

    out = {
        "task_id": item.get("task_id", ""),
        "instruction": str(prompt),
        "prompt": str(prompt),
        "response": str(code),
        "reference_code": str(code),
        "test_setup_code": str(test_setup_code or ""),
        "test_imports": test_imports if test_imports is not None else [],
        "test_list": test_list,
        "tests": test_list,
        "test": "\n".join(test_list),
        "challenge_test_list": challenge_test_list,
        "benchmark": "mbpp",
        "config": config,
        "split": split,
    }

    if entry_point:
        out["entry_point"] = str(entry_point)

    for k in ["source_file", "text", "code"]:
        if k in item and k not in out:
            out[k] = item[k]

    return out


def download_mbpp(args) -> List[Dict[str, Any]]:
    ds = load_dataset(
        "Muennighoff/mbpp",
        args.config,
        split=args.split,
        trust_remote_code=True,
    )

    rows = []
    with_test = 0
    with_entry_point = 0

    for item in ds:
        row = normalize_mbpp_record(
            item=item,
            config=args.config,
            split=args.split,
        )

        if row.get("test_list"):
            with_test += 1
        if row.get("entry_point"):
            with_entry_point += 1

        rows.append(row)

    print("=" * 80)
    print("[MBPP Download Summary]")
    print("=" * 80)
    print(f"Dataset              : Muennighoff/mbpp")
    print(f"Config               : {args.config}")
    print(f"Split                : {args.split}")
    print(f"Saved rows           : {len(rows)}")
    print(f"Rows with tests      : {with_test}/{len(rows)}")
    print(f"Rows with entry_point: {with_entry_point}/{len(rows)}")
    if rows:
        print(f"Example output keys  : {list(rows[0].keys())}")

    return rows


# ============================================================
# Args
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="kodcode",
        choices=["kodcode", "mbpp"],
        help="kodcode = training data, mbpp = evaluation data",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file path.",
    )

    # KodCode args
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="KodCode/KodCode-V1-SFT-R1",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="For KodCode default=train. For MBPP default=test.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples for KodCode. Ignored for MBPP unless you modify manually.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--shuffle-buffer-size", type=int, default=50000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=1000)

    # MBPP args
    parser.add_argument(
        "--config",
        type=str,
        default="sanitized",
        choices=["sanitized", "full"],
        help="MBPP config.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.split is None:
        if args.source == "kodcode":
            args.split = "train"
        elif args.source == "mbpp":
            args.split = "test"

    if args.output_file is None:
        if args.source == "kodcode":
            args.output_file = f"kodcode_v1_sft_r1_sample_{args.sample_size}.jsonl"
        elif args.source == "mbpp":
            args.output_file = f"mbpp_{args.config}_{args.split}.jsonl"

    if args.source == "kodcode":
        rows = download_kodcode(args)
    elif args.source == "mbpp":
        rows = download_mbpp(args)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    write_jsonl(rows, args.output_file)

    print("=" * 80)
    print(f"Saved {len(rows)} rows to {args.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
