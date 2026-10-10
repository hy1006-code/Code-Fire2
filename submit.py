import argparse
import csv
from pathlib import Path

import torch

from test import SUPPORTED_SUFFIXES, load_model, predict_image


PROJECT_ROOT = Path(__file__).resolve().parent


def write_submission(predictions: dict[str, int], output_path: Path) -> None:
    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "label"])
        for name in sorted(predictions):
            writer.writerow([name, predictions[name]])


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Kaggle 火情识别提交文件")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "best_checkpoint.pt")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "data" / "test" / "images")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    predictions = {}
    for image_path in sorted(args.image_dir.iterdir()):
        if image_path.suffix.lower() in SUPPORTED_SUFFIXES:
            prediction, _ = predict_image(model, image_path, device)
            predictions[image_path.name] = prediction

    write_submission(predictions, args.output)
    print(f"saved_submission={args.output.resolve()} count={len(predictions)}")


if __name__ == "__main__":
    main()
