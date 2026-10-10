"""火灾识别 Web 演示（Streamlit）。

基于训练好的 FireCNN 二分类模型，提供两种识别方式：
1. 上传图片识别：上传本地图片，展示火情概率与判定结果；
2. 摄像头实时识别：调用本机摄像头，逐帧推理并叠加判定文字。

运行方式：
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from model import FireCNN
from test import PROJECT_ROOT, SUPPORTED_SUFFIXES, load_model
from train import build_transform

# 页面标题必须最先调用
st.set_page_config(page_title="火灾识别", page_icon="🔥", layout="wide")

CHECKPOINTS = {
    "aug_best.pt（数据增强训练，验证集最优）": "aug_best.pt",
    "ft_best.pt（难负样本微调）": "ft_best.pt",
    "best_checkpoint.pt（初版训练）": "best_checkpoint.pt",
}


def get_device() -> torch.device:
    """按优先级选择推理设备：CUDA -> MPS -> CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner="正在加载模型...")
def get_model(checkpoint: str, device: torch.device) -> FireCNN:
    """按 checkpoint 路径缓存模型，避免每次交互重复加载。"""
    model = load_model(PROJECT_ROOT / checkpoint, device)
    return model


@torch.no_grad()
def predict_pil(model: FireCNN, image: Image.Image, device: torch.device) -> float:
    """对 PIL 图片做预处理并返回火情概率（0~1）。"""
    tensor = build_transform(train=False)(image).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)
    return probabilities[0, 1].item()


def verdict_banner(prob: float, threshold: float) -> None:
    """按阈值展示判定结果横幅，并按置信度给出三档提示。"""
    if prob >= threshold:
        if prob >= 0.7:
            st.error(f"🔥 检测到火灾（高置信度，概率 {prob:.1%}）")
        else:
            st.warning(f"⚠️ 疑似火情，请人工确认（概率 {prob:.1%}）")
    else:
        st.success(f"✅ 未见火灾（概率 {prob:.1%}）")


def show_prediction(image: Image.Image, prob: float, threshold: float) -> None:
    """展示图片、概率条与判定结果。"""
    st.image(image, caption="待识别图片", use_container_width=True)
    st.progress(min(prob, 1.0), text=f"火情概率：{prob:.1%}")
    verdict_banner(prob, threshold)


def upload_tab(model: FireCNN, device: torch.device, threshold: float) -> None:
    """上传图片识别页签。"""
    st.subheader("📷 上传图片识别")
    uploaded = st.file_uploader(
        "选择图片（jpg / jpeg / png）", type=["jpg", "jpeg", "png"]
    )
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        show_prediction(image, predict_pil(model, image, device), threshold)

    st.divider()
    st.caption("或者从验证集选一张示例图快速体验：")
    sample_dir = PROJECT_ROOT / "data" / "validation" / "images"
    samples = sorted(p.name for p in sample_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
    sample = st.selectbox("示例图片", samples, label_visibility="collapsed")
    if st.button("识别这张示例图", type="primary"):
        image = Image.open(sample_dir / sample).convert("RGB")
        show_prediction(image, predict_pil(model, image, device), threshold)


def webcam_tab(model: FireCNN, device: torch.device, threshold: float) -> None:
    """摄像头实时识别页签（依赖 streamlit-webrtc）。"""
    st.subheader("🎥 摄像头实时识别")

    try:
        import av
        import cv2
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("未安装 streamlit-webrtc / av，无法使用摄像头。请先 `pip install streamlit-webrtc av`。")
        return

    transform = build_transform(train=False)

    class FireVideoProcessor(VideoProcessorBase):
        """逐帧推理：把摄像头画面缩放到 128x128 送入模型，叠加判定文字。"""

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            bgr = frame.to_ndarray(format="bgr24")
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            tensor = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1)[0, 1].item()
            if prob >= threshold:
                text, color = f"FIRE {prob:.0%}", (0, 0, 255)
            else:
                text, color = f"SAFE {prob:.0%}", (0, 180, 0)
            cv2.rectangle(bgr, (8, 8), (320, 64), color, -1)
            cv2.putText(bgr, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            return av.VideoFrame.from_ndarray(bgr, format="bgr24")

    webrtc_streamer(
        key="fire-webcam",
        video_processor_factory=FireVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("浏览器会弹出摄像头授权请求；关闭开关即停止识别。")


def main() -> None:
    st.title("🔥 火灾识别系统")
    st.markdown("基于 **FireCNN** 卷积神经网络的火灾图像二分类演示")

    device = get_device()
    with st.sidebar:
        st.header("⚙️ 参数设置")
        checkpoint = st.selectbox("模型权重", list(CHECKPOINTS))
        threshold = st.slider("判定阈值（越低越敏感）", 0.0, 1.0, 0.5, 0.05)
        st.caption(f"当前推理设备：**{device}**")
        st.caption("提示：模型对火情概率输出偏保守，漏检时可调低阈值"
                   "（验证集上 F1 最优阈值约 0.05）。")

    model = get_model(CHECKPOINTS[checkpoint], device)

    tab_upload, tab_webcam = st.tabs(["📷 上传图片", "🎥 摄像头实时"])
    with tab_upload:
        upload_tab(model, device, threshold)
    with tab_webcam:
        webcam_tab(model, device, threshold)


if __name__ == "__main__":
    main()
