"""Compare the vest YOLO model with JEV alone on one image-level question.

Ground truth: an image is positive when its label file contains any
``NO_Safety_Vest`` box. YOLO is positive when it detects that class; JEV
(``vision="jev"``) is positive when its probability reaches ``--threshold``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "vest_dataset"
YOLO_WEIGHTS = ROOT / "model" / "best_bymandy.pt"
NO_VEST_CLASS = 0
NO_VEST_NAME = "NO_Safety_Vest"
QUESTION = "Is any person in this image not wearing a safety vest?"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ground_truth(label_path: Path) -> bool:
    if not label_path.is_file():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and int(float(parts[0])) == NO_VEST_CLASS:
            return True
    return False


def run_yolo(images: list[Path], weights: Path, confidence: float) -> dict[Path, dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {int(key): str(value) for key, value in model.names.items()}
    no_vest_ids = {key for key, value in names.items() if value == NO_VEST_NAME}
    if not no_vest_ids:
        raise ValueError(f"{weights} has no {NO_VEST_NAME} class: {names}")
    outputs = {}
    for path in images:
        started = time.perf_counter()
        result = model.predict(source=str(path), conf=confidence, verbose=False)[0]
        elapsed = time.perf_counter() - started
        scores = [
            float(score)
            for cls, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
            if int(cls) in no_vest_ids
        ]
        outputs[path] = {
            "score": max(scores, default=0.0),
            "predicted": bool(scores),
            "seconds": elapsed,
        }
    return outputs


def run_jev(images: list[Path], threshold: float) -> dict[Path, dict]:
    from jev_sees import Sees

    sees = Sees(vision="jev")
    outputs = {}
    for path in images:
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = sees.probability(str(path), QUESTION)
            probability = float(next(iter(result.answers.values())).noul)
        except Exception as exc:  # noqa: BLE001 - one failed request must not end the run
            outputs[path] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        outputs[path] = {
            "score": probability,
            "predicted": probability >= threshold,
            "seconds": time.perf_counter() - started,
        }
    return outputs


def metrics(truth: dict[Path, bool], outputs: dict[Path, dict]) -> dict[str, float]:
    tp = fp = fn = tn = 0
    seconds = []
    for path, output in outputs.items():
        if "error" in output:
            continue
        seconds.append(output["seconds"])
        actual, predicted = truth[path], output["predicted"]
        tp += actual and predicted
        fp += (not actual) and predicted
        fn += actual and not predicted
        tn += (not actual) and not predicted
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": total,
        "errors": len(outputs) - total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sec/img": sum(seconds) / len(seconds) if seconds else 0.0,
    }


def print_table(results: dict[str, dict[str, float]]) -> None:
    columns = list(next(iter(results.values())))
    print("method  " + "  ".join(f"{name:>9}" for name in columns))
    for method, row in results.items():
        cells = []
        for name in columns:
            value = row[name]
            cells.append(f"{value:>9}" if isinstance(value, int) else f"{value:>9.3f}")
        print(f"{method:<6}  " + "  ".join(cells))


def write_csv(
    path: Path,
    images: list[Path],
    truth: dict[Path, bool],
    runs: dict[str, dict[Path, dict]],
) -> None:
    fields = ["image", "truth"]
    for method in runs:
        fields += [f"{method}_score", f"{method}_predicted", f"{method}_error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in images:
            row = {"image": image.name, "truth": int(truth[image])}
            for method, outputs in runs.items():
                output = outputs.get(image, {})
                row[f"{method}_score"] = (
                    f"{output['score']:.4f}" if "score" in output else ""
                )
                row[f"{method}_predicted"] = (
                    int(output["predicted"]) if "predicted" in output else ""
                )
                row[f"{method}_error"] = output.get("error", "")
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument("--methods", nargs="+", choices=["yolo", "jev"], default=["yolo", "jev"])
    parser.add_argument("--weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5, help="JEV probability cutoff")
    parser.add_argument("--csv", type=Path, default=ROOT / "vest_yolo_vs_jev.csv")
    args = parser.parse_args()

    image_dir = DATASET / args.split / "images"
    label_dir = DATASET / args.split / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")
    truth = {path: ground_truth(label_dir / f"{path.stem}.txt") for path in images}
    positives = sum(truth.values())
    print(f"{len(images)} images, {positives} with {NO_VEST_NAME}, {len(images) - positives} without")

    runs: dict[str, dict[Path, dict]] = {}
    if "yolo" in args.methods:
        runs["yolo"] = run_yolo(images, args.weights, args.yolo_conf)
    if "jev" in args.methods:
        runs["jev"] = run_jev(images, args.threshold)

    print_table({method: metrics(truth, outputs) for method, outputs in runs.items()})
    write_csv(args.csv, images, truth, runs)
    print(f"Per-image results: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
