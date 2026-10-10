"""在官方测试集 data/test（有标签）上评估新旧模型，比较 F1 并扫描阈值。"""
import json

from test import PROJECT_ROOT, load_model
from tune_threshold import f1_score, fire_probabilities, select_device, sweep


def main() -> None:
    device = select_device()
    print(f"device={device}")

    test_labels = json.loads((PROJECT_ROOT / "data" / "test" / "labels.json").read_text(encoding="utf-8"))

    for tag, ckpt in [("old-best", "best_checkpoint.pt"), ("aug-best", "aug_best.pt"), ("aug-latest", "aug_latest.pt")]:
        model = load_model(PROJECT_ROOT / ckpt, device)
        probs = fire_probabilities(model, PROJECT_ROOT / "data" / "test" / "images", device)
        f1_050 = f1_score(test_labels, probs, 0.50)[0]
        print(f"\n=== {tag} ({ckpt}) F1@0.5 = {f1_050:.4f} ===")
        sweep(test_labels, probs)


if __name__ == "__main__":
    main()
