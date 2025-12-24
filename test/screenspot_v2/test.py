import argparse
import logging

from evaluator import run
from tester.qwen3vl_tester import Qwen3VLTester


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tester", type=str, required=True, help="Tester name, e.g., qwen3vl")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="mobile|desktop|web|all")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.tester.lower() == "qwen3vl":
        adapter = Qwen3VLTester(args.model_path)
    else:
        raise ValueError(f"Unknown tester: {args.tester}")

    out = run(adapter, args.dataset_path, args.task, args.output)
    logging.info("Tasks Result: %s", out["tasks_result"])


if __name__ == "__main__":
    main()
