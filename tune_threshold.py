"""阈值调优：在本地带标签的测试/验证集上扫描分类阈值，选取 F1 最优阈值，再对 val2017 重新预测并生成提交文件。"""
import csv
import json
from pathlib import Path

import torch

from test import SUPPORTED_SUFFIXES, PROJECT_ROOT, load_model, predict_image


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def fire_probabilities(model, image_dir: Path, device: torch.device) -> dict[str, float]:
    """对目录下所有图片计算有火概率（不阈值化）。"""
    probs = {}
    for image_path in sorted(Path(image_dir).iterdir()):
        if image_path.suffix.lower() in SUPPORTED_SUFFIXES:
            _, fire_probability = predict_image(model, image_path, device)
            probs[image_path.name] = fire_probability
    return probs


def f1_score(labels: dict[str, int], probs: dict[str, float], threshold: float) -> tuple[float, int, int, int]:
    names = sorted(set(labels) & set(probs))
    y_true = [labels[n] for n in names]
    y_pred = [int(probs[n] >= threshold) for n in names]
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, tp, fp, fn


def sweep(labels: dict[str, int], probs: dict[str, float]) -> None:
    best = (0.0, 0.5)
    for t in [i / 100 for i in range(50, 100)]:
        f1, tp, fp, fn = f1_score(labels, probs, t)
        if f1 > best[0]:
            best = (f1, t)
        print(f"thr={t:.2f} f1={f1:.4f} tp={tp} fp={fp} fn={fn}")
    print(f"BEST: threshold={best[1]:.2f} f1={best[0]:.4f}")


def write_submissions(predictions: dict[str, int], output_csv: Path, output_json: Path) -> None:
    with Path(output_csv).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        for name in sorted(predictions):
            writer.writerow([name, predictions[name]])
    with Path(output_json).open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="只对 val2017 重新预测并写提交文件（不扫阈值）")
    args = parser.parse_args()

    device = select_device()
    print(f"device={device}")
    model = load_model(PROJECT_ROOT / "best_checkpoint.pt", device)

    if args.threshold is not None:
        val2017_probs = fire_probabilities(model, PROJECT_ROOT / "val2017", device)
        predictions = {name: int(prob >= args.threshold) for name, prob in val2017_probs.items()}
        write_submissions(predictions, PROJECT_ROOT / "submission.csv", PROJECT_ROOT / "submission.json")
        n_fire = sum(predictions.values())
        print(f"submission saved: {len(predictions)} images, {n_fire} fire, {len(predictions) - n_fire} no-fire")
        return

    test_labels = json.loads((PROJECT_ROOT / "data" / "test" / "labels.json").read_text(encoding="utf-8"))
    test_probs = fire_probabilities(model, PROJECT_ROOT / "data" / "test" / "images", device)
    print("=== test set threshold sweep ===")
    sweep(test_labels, test_probs)

    val_labels = json.loads((PROJECT_ROOT / "data" / "validation" / "labels.json").read_text(encoding="utf-8"))
    val_probs = fire_probabilities(model, PROJECT_ROOT / "data" / "validation" / "images", device)
    print("=== validation set threshold sweep ===")
    sweep(val_labels, val_probs)


if __name__ == "__main__":
    main()
