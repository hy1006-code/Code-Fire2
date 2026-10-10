import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from model import FireCNN
from train import build_transform, classification_metrics


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_model(path: Path, device: torch.device) -> FireCNN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FireCNN()
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: Path, device: torch.device) -> tuple[int, float]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    tensor = build_transform(train=False)(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)
    fire_probability = probabilities[0, 1].item()
    return int(fire_probability >= 0.5), fire_probability


def predict_directory(model, image_dir: Path, output_path: Path, device: torch.device) -> dict[str, int]:
    predictions = {}
    for image_path in sorted(Path(image_dir).iterdir()):
        if image_path.suffix.lower() in SUPPORTED_SUFFIXES:
            prediction, _ = predict_image(model, image_path, device)
            predictions[image_path.name] = prediction
    Path(output_path).write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return predictions


def print_metrics(predictions: dict[str, int], labels_path: Path) -> None:
    with Path(labels_path).open("r", encoding="utf-8") as file:
        labels = {name: int(label) for name, label in json.load(file).items()}
    names = sorted(set(labels) & set(predictions))
    metrics = classification_metrics(
        [labels[name] for name in names], [predictions[name] for name in names]
    )
    accuracy = sum(labels[name] == predictions[name] for name in names) / len(names)
    print(
        f"samples={len(names)} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} accuracy={accuracy:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="火情二分类推理")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "best_checkpoint.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "result.json")
    parser.add_argument("--image", type=Path, help="单张图片推理路径")
    parser.add_argument("--labels", type=Path, help="本地评估标签路径")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    model = load_model(args.checkpoint, device)

    if args.image:
        prediction, confidence = predict_image(model, args.image, device)
        print(f"image={args.image.name} prediction={prediction} fire_probability={confidence:.4f}")
        return

    predictions = predict_directory(model, args.data_dir / "images", args.output, device)
    print(f"saved_predictions={args.output.resolve()} count={len(predictions)}")
    labels_path = args.labels or args.data_dir / "labels.json"
    if labels_path.exists():
        print_metrics(predictions, labels_path)


if __name__ == "__main__":
    main()
