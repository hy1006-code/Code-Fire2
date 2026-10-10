# Code-Fire2 — 火灾识别（CNN 二分类）

基于 PyTorch 的卷积神经网络火灾图像识别工程，包含数据准备、训练、微调、阈值调优、推理提交与 Streamlit 演示网站。

## 模型

`FireCNN`：3 组 conv+relu+maxpool（16→32→64 通道），输入 128×128 RGB 图像，
经 fc(16384→256)→dropout→fc(256→2) 输出火/非火二分类概率。见 `model.py`。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `train.py` | 主训练脚本（含数据增强、验证、早停选择最优 checkpoint） |
| `ft_hard.py` | 难负样本微调，按官方测试集 F1 选择最优 checkpoint |
| `tune_threshold.py` | 判定阈值扫描（0.05~0.95），按 F1 选最优阈值 |
| `test.py` | 推理：单图/目录预测、本地评估指标 |
| `submit.py` | 生成竞赛提交文件 |
| `eval_aug.py` | 数据增强训练结果的评估 |
| `app.py` | Streamlit 演示网站（上传图片 + 摄像头实时识别） |
| `aug_best.pt` | 数据增强训练的最优权重（验证集最优） |
| `ft_best.pt` | 难负样本微调的最优权重 |
| `best_checkpoint.pt` | 初版训练的最优权重 |

## 环境

```bash
pip install -r requirements.txt
# 还需 PyTorch，按官方指引安装
```

## 使用

```bash
# 单张图片推理
python test.py --checkpoint aug_best.pt --image 路径/图片.jpg

# 启动演示网站（支持上传图片与摄像头实时识别）
streamlit run app.py
```

## 备注

- 模型对火情概率输出偏保守，验证集上 F1 最优阈值约 0.05，演示时可调低阈值提高召回。
- 数据集（`data/`、`val2017/` 等）体积较大未上传；训练需自备数据并按其目录结构放置。
