import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS5_ROOT = PROJECT_ROOT.parent
VALIDATION_PREFIX = "20250526_firesmoke_"


def load_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as file:
        return {name: int(label) for name, label in json.load(file).items()}


def check_images_exist(image_dir: Path, labels: dict[str, int]) -> None:
    for name in labels:
        if not (image_dir / name).is_file():
            raise FileNotFoundError(f"标注对应的图像不存在: {image_dir / name}")


def copy_split(image_dir: Path, labels: dict[str, int], destination: Path) -> None:
    destination_images = destination / "images"
    destination_images.mkdir(parents=True)
    for name in labels:
        shutil.copy2(image_dir / name, destination_images / name)
    (destination / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_data(
    output_dir: Path,
    train_images: Path,
    train_labels: Path,
    test_images: Path,
    test_labels: Path,
) -> dict[str, int]:
    """Copy source data into a leakage-safe train/validation/test tree."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output_dir}")

    all_train_labels = load_labels(Path(train_labels))
    test_label_map = load_labels(Path(test_labels))
    validation_labels = {
        name: label
        for name, label in all_train_labels.items()
        if name.startswith(VALIDATION_PREFIX)
    }
    train_label_map = {
        name: label
        for name, label in all_train_labels.items()
        if not name.startswith(VALIDATION_PREFIX)
    }
    if not validation_labels:
        raise ValueError(f"未找到验证集前缀: {VALIDATION_PREFIX}")

    check_images_exist(Path(train_images), all_train_labels)
    check_images_exist(Path(test_images), test_label_map)

    output_dir.mkdir()
    copy_split(Path(train_images), train_label_map, output_dir / "train")
    copy_split(Path(train_images), validation_labels, output_dir / "validation")
    copy_split(Path(test_images), test_label_map, output_dir / "test")

    return {
        "train": len(train_label_map),
        "validation": len(validation_labels),
        "test": len(test_label_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="整理火灾检测课程数据集")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    counts = prepare_data(
        args.output,
        CLASS5_ROOT / "火灾检测" / "images",
        CLASS5_ROOT / "火灾检测" / "train_image.json",
        CLASS5_ROOT / "test" / "test",
        CLASS5_ROOT / "test" / "test_labels.json",
    )
    print("数据整理完成:", ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
