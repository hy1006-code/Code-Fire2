import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import FireCNN


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 128


class FireDataset(Dataset):
    def __init__(self, split_dir: Path, transform):
        self.image_dir = Path(split_dir) / "images"
        self.transform = transform
        with (Path(split_dir) / "labels.json").open("r", encoding="utf-8") as file:
            labels = json.load(file)
        self.samples = sorted((name, int(label)) for name, label in labels.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, label = self.samples[index]
        with Image.open(self.image_dir / name) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def build_transform(train: bool):
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_history(history_path: Path, history: list[dict]) -> None:
    Path(history_path).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_f1: float,
    history: list[dict],
    seed: int,
    validation: dict | None = None,
    rng_state: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "history": history,
            "seed": seed,
            "validation": validation,
            "rng_state": rng_state,
        },
        checkpoint_path,
    )


def load_training_checkpoint(
    checkpoint_path: Path, model: nn.Module, optimizer: optim.Optimizer
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def get_rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state(),
    }


def restore_rng_state(rng_state: dict | None, generator: torch.Generator) -> None:
    if rng_state is None:
        return
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])
    generator.set_state(rng_state["loader"])


def classification_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels, predictions = [], []
    for images, batch_labels in loader:
        images, batch_labels = images.to(device), batch_labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, batch_labels).item() * len(batch_labels)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="火情二分类基准模型训练")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "best_checkpoint.pt")
    parser.add_argument(
        "--latest-checkpoint", type=Path, default=PROJECT_ROOT / "latest_checkpoint.pt"
    )
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "history.json")
    parser.add_argument("--resume", type=Path, help="从 latest_checkpoint.pt 等可信断点继续训练")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}, seed={args.seed}")

    train_set = FireDataset(args.data_dir / "train", build_transform(train=True))
    validation_set = FireDataset(args.data_dir / "validation", build_transform(train=False))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_set, args.batch_size, shuffle=False, num_workers=0)

    model = FireCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_f1 = -1.0
    history = []
    start_epoch = 1

    if args.resume is not None:
        if not args.resume.is_file():
            parser.error(f"未找到断点：{args.resume}")
        checkpoint = load_training_checkpoint(args.resume, model, optimizer)
        best_f1 = float(checkpoint["best_f1"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"), generator)
        print(f"resumed_from={args.resume.resolve()}, start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

        validation = evaluate(model, validation_loader, criterion, device)
        train_loss = running_loss / len(train_set)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_precision": validation["precision"],
            "val_recall": validation["recall"],
            "val_f1": validation["f1"],
        }
        history.append(record)
        save_history(args.history, history)
        is_best = validation["f1"] > best_f1
        if is_best:
            best_f1 = validation["f1"]
        rng_state = get_rng_state(generator)
        save_training_checkpoint(
            args.latest_checkpoint,
            model,
            optimizer,
            epoch,
            best_f1,
            history,
            args.seed,
            validation,
            rng_state,
        )
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"val_loss={validation['loss']:.4f} val_precision={validation['precision']:.4f} "
            f"val_recall={validation['recall']:.4f} val_f1={validation['f1']:.4f}"
        )
        if is_best:
            save_training_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch,
                best_f1,
                history,
                args.seed,
                validation,
                rng_state,
            )
            print(f"saved_best_checkpoint={args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
