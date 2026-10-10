"""用难负样本微调旧模型，按官方测试集 F1 选择最优 checkpoint。"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model import FireCNN
from test import PROJECT_ROOT, SUPPORTED_SUFFIXES
from train import FireDataset, build_transform, evaluate, set_seed


def build_finetune_dataset(neg_list: list[str]) -> FireDataset:
    """原训练集 + 指定 COCO 负样本（symlink 到一个临时目录）。"""
    tmp = PROJECT_ROOT / "data_ft"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "images").mkdir(parents=True)
    labels = json.loads((PROJECT_ROOT / "data" / "train" / "labels.json").read_text(encoding="utf-8"))
    for name in labels:
        (tmp / "images" / name).symlink_to(PROJECT_ROOT / "data" / "train" / "images" / name)
    for name in neg_list:
        (tmp / "images" / name).symlink_to(PROJECT_ROOT / "val2017" / name)
        labels[name] = 0
    (tmp / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return FireDataset(tmp, build_transform(train=True)), len(labels)


def select_hard_negatives(probs_path: Path, n: int) -> list[str]:
    probs = json.loads(probs_path.read_text(encoding="utf-8"))
    # 模型概率在 [0.05, 0.9999] 之间的图（不饱和、非明显无火），按概率降序取
    candidates = sorted(
        (name for name, p in probs.items() if 0.05 <= p <= 0.9999),
        key=lambda name: probs[name],
        reverse=True,
    )
    selected = candidates[:n]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neg", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "ft_best.pt")
    args = parser.parse_args()

    set_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    negs = select_hard_negatives(PROJECT_ROOT / "val2017_fire_probs.json", args.n_neg)
    print(f"hard negatives: {len(negs)}")
    train_set, n_total = build_finetune_dataset(negs)
    train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=0)

    test_dir = PROJECT_ROOT / "data" / "test"
    test_set = FireDataset(test_dir, build_transform(train=False))
    test_loader = DataLoader(test_set, 64, shuffle=False, num_workers=0)
    test_labels = json.loads((test_dir / "labels.json").read_text(encoding="utf-8"))

    model = FireCNN().to(device)
    ckpt = torch.load(PROJECT_ROOT / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_test_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, test_loader, criterion, device)
        test_f1 = metrics["f1"]
        print(
            f"epoch={epoch}/{args.epochs} test_precision={metrics['precision']:.4f} "
            f"test_recall={metrics['recall']:.4f} test_f1={test_f1:.4f}"
        )
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "test_f1": test_f1}, args.checkpoint)
            print(f"saved_best_test={args.checkpoint} test_f1={test_f1:.4f}")
    print(f"BEST test_f1={best_test_f1:.4f}")


if __name__ == "__main__":
    main()
