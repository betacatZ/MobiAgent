import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm
from tester import Qwen3VLTester


def _normalize_bbox(bbox: List[float], img_size: Tuple[int, int]) -> List[float]:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    W, H = img_size
    return [x / W, y / H, x2 / W, y2 / H]


def evaluate(tester, dataset_path: str, task: str) -> Dict:
    tasks = ["mobile", "desktop", "web"] if task == "all" else [task]

    tasks_result: List[List[float]] = []
    results: List[Dict] = []

    for t in tasks:
        dataset_file = os.path.join(dataset_path, f"screenspot_{t}_v2.json")
        with open(dataset_file, "r") as f:
            screenspot_data = json.load(f)

        num_action = 0
        corr_action = 0
        text_correct: List[int] = []
        icon_correct: List[int] = []
        num_wrong_format = 0

        for j, item in tqdm(enumerate(screenspot_data), total=len(screenspot_data)):
            num_action += 1

            filename = item["img_filename"]
            img_path = os.path.join(dataset_path, "screenspotv2_image", filename)
            if not os.path.exists(img_path):
                logging.info("img not found: %s", img_path)
                num_wrong_format += 1
                if item["data_type"] == "text":
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                continue

            image = Image.open(img_path)
            instruction = item["instruction"]
            bbox_norm = _normalize_bbox(item["bbox"], image.size)

            click_point: Optional[Tuple[float, float]] = tester.generate_click_coordinate(instruction, image)
            if click_point is None:
                num_wrong_format += 1
                if item["data_type"] == "text":
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                tqdm.write("Step: %s wrong format" % str(j))
                continue

            x1, y1, x2, y2 = bbox_norm
            if x1 <= click_point[0] <= x2 and y1 <= click_point[1] <= y2:
                corr_action += 1
                if item["data_type"] == "text":
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                tqdm.write("match %.6f" % (corr_action / max(num_action, 1)))
            else:
                if item["data_type"] == "text":
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                tqdm.write("unmatch %.6f" % (corr_action / max(num_action, 1)))

            results.append(
                {
                    "img_path": img_path,
                    "text": instruction,
                    "bbox": bbox_norm,
                    "pred": [click_point[0], click_point[1]],
                    "type": item["data_type"],
                    "source": item["data_source"],
                }
            )

        logging.info("Action Acc: %.6f", corr_action / max(num_action, 1))
        logging.info("Total num: %d", num_action)
        logging.info("Wrong format num: %d", num_wrong_format)
        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0.0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0.0
        logging.info("Text Acc: %.6f", text_acc)
        logging.info("Icon Acc: %.6f", icon_acc)

        tasks_result.append([text_acc, icon_acc])

    return {"tasks_result": tasks_result, "results": results}


def run(tester, model_path: str, dataset_path: str, task: str, output: str) -> Dict:
    out = evaluate(tester, dataset_path, task)
    model = os.path.basename(model_path)
    output_json = os.path.join(output, model, f"screenspot_v2_{task}_results.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f)
    return out


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model name, e.g., qwen3vl")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="mobile|desktop|web|all")
    parser.add_argument("--output", type=str, default="./output")
    args = parser.parse_args()

    if args.model.lower() == "qwen3vl":
        tester = Qwen3VLTester(args.model_path)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    out = run(tester, args.model_path, args.dataset_path, args.task, args.output)
    logging.info("Tasks Result: %s", out["tasks_result"])


if __name__ == "__main__":
    main()
