"""
Diffusion backend - STUB 版

扩散模型调用全部用 PIL 占位图,用于:
  - 本地无 GPU 验证 pipeline 接口
  - 离线 dry run
  - 改动 χ / ζ 时的快速 e2e 回归

【接口契约】(real backend 也必须遵守同一份契约)
  call_sd(prompt, ...)       → 1 张 PIL.Image
  call_mv_adapter(ref)       → 6 张 PIL.Image, 顺序对应 azimuth [0, 60, 120, 180, 240, 300]
  call_hunyuan3d(ref)        → 2 张 PIL.Image, 顺序对应 azimuth [135, 225]

视角顺序是 backend 和 cinematographer 之间的契约。cinematographer 的 VIEWS 表
按这个顺序声明 mv_adapter 和 hunyuan3d 来源的 view,zip 时位置一定对得上。
"""

from typing import Optional

from PIL import Image, ImageDraw, ImageFont


# 视角顺序契约 (real backend 也要遵守同一份)
# MV-Adapter SDXL 的实际 azimuth 顺序,从 scripts/inference_i2mv_sdxl.py 读出
MV_ADAPTER_AZIMUTHS = [0, 45, 90, 180, 270, 315]
HUNYUAN3D_AZIMUTHS = [135, 225]  # 占位,简版不用


def _placeholder(w: int, h: int, label: str) -> Image.Image:
    """白底 + 中央椭圆 + 标签文字。

    椭圆的作用是让下游 rembg / 白底兜底抠图能扫出一个非空的 bbox,
    这样 cinematographer 的 _compute_bbox 才能算出 char_height_px,
    ζ 才能正确缩放 — 整个 stub e2e 链路都依赖这个椭圆。
    """
    img = Image.new("RGB", (w, h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    mx, my = w // 4, h // 6
    draw.ellipse([mx, my, w - mx, h - my], fill=(120, 120, 130))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    lines = [label[i:i + 50] for i in range(0, len(label), 50)][:6]
    y = h - 140
    for line in lines:
        draw.text((20, y), line, fill=(20, 20, 20), font=font)
        y += 22
    return img


def call_sd(prompt: str, negative: str = "",
            control_image: Optional[Image.Image] = None,
            w: int = 768, h: int = 1024) -> Image.Image:
    """STUB SD1.5 调用。real backend 走 diffusers + SD1.5 + ControlNet OpenPose。"""
    return _placeholder(w, h, f"SD: {prompt[:100]}")


def call_mv_adapter(ref: Image.Image, prompt: str = "") -> list[Image.Image]:
    """STUB MV-Adapter。返回 6 张图,尺寸跟 ref 一致,顺序按 MV_ADAPTER_AZIMUTHS。

    第二个参数 prompt 是为了对齐 real backend 的签名。stub 不用。
    """
    w, h = ref.size
    return [_placeholder(w, h, f"MV az={az}") for az in MV_ADAPTER_AZIMUTHS]


def call_hunyuan3d(ref: Image.Image) -> list[Image.Image]:
    """STUB Hunyuan3D-1 stage 1。返回 2 张图,顺序按 HUNYUAN3D_AZIMUTHS。"""
    w, h = ref.size
    return [_placeholder(w, h, f"H3D az={az}") for az in HUNYUAN3D_AZIMUTHS]
