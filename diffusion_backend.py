"""
Diffusion backend - REAL 版

实现:
  call_sd          → diffusers SD1.5 + ControlNet OpenPose,A-pose 骨架自动生成
  call_mv_adapter  → subprocess 调 MV-Adapter SDXL i2mv,切 grid 成 6 张
  call_hunyuan3d   → 仍 stub (双人简版用不到,保持接口签名)

接口契约 (跟 stub 一致):
  call_sd(prompt, ...)               → 1 张 PIL.Image
  call_mv_adapter(ref, prompt)       → 6 张 PIL.Image,顺序按 MV_ADAPTER_AZIMUTHS

依赖:
  - torch + diffusers + transformers
  - MV-Adapter clone 在 MV_ADAPTER_REPO 路径下,且其 demo 已跑通

性能优化:
  - SD pipeline lazy load 一次,缓存全局
  - A-pose 骨架生成一次,缓存到 SKELETON_CACHE
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw


# ---------- 配置 ----------

# MV-Adapter clone 路径(必须先 git clone 并跑通 demo)
MV_ADAPTER_REPO = Path(os.environ.get("MV_ADAPTER_REPO", "/root/MV-Adapter"))

# 用哪个 Python 解释器跑 MV-Adapter subprocess (默认就是当前的)
MV_ADAPTER_PYTHON = os.environ.get("MV_ADAPTER_PYTHON", sys.executable)

# A-pose 骨架缓存文件 (生成一次,后续所有角色复用同一张)
SKELETON_CACHE = Path(os.environ.get(
    "DD_SKELETON_CACHE", "/root/dd_outputs/_assets/apose_skeleton.png"
))

# SD1.5 角色生成尺寸 (头肩证件照构图,跟 cinematographer.py 的 CHAR_IMG_W/H 对齐)
SD_CHAR_W, SD_CHAR_H = 768, 768

# MV-Adapter SDXL 输出 tile 尺寸
MV_TILE_SIZE = 768

# MV-Adapter 6 视角的 azimuth 顺序 (从 MV-Adapter 源码 line 134 读出)
# 0° = 正面, 顺时针递增
MV_ADAPTER_AZIMUTHS = [0, 45, 90, 180, 270, 315]
HUNYUAN3D_AZIMUTHS = [135, 225]  # 占位,我们不用


# ---------- 全局 lazy state ----------

_sd_pipe = None


def _get_sd_pipe():
    """Lazy load SD1.5 + ControlNet OpenPose pipeline. 只 from_pretrained 一次。"""
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe

    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler,
    )

    print("[Γ-real] 加载 SD1.5 + ControlNet OpenPose (首次,后续复用)...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_vae_slicing()

    _sd_pipe = pipe
    print("[Γ-real] SD pipeline 就绪")
    return _sd_pipe


# ---------- A-pose 骨架生成 ----------

def _build_apose_skeleton(width: int = SD_CHAR_W, height: int = SD_CHAR_H) -> Image.Image:
    """合成一张极简的 OpenPose 头肩骨架。

    只画 5 个关键点: 鼻、颈、双肩、外加双肩中点。
    不画手肘/手腕/髋,因为我们想要的是头肩近景(证件照构图),不要躯干和手。

    位置校准:
      - 鼻在画布 30% (头部中心,留出顶部头发空间)
      - 颈在 55%
      - 肩在 65% (画面下边缘附近,模拟"画面切到肩部下方"的构图)
    """
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = width // 2
    y_nose     = int(height * 0.30)
    y_neck     = int(height * 0.55)
    y_shoulder = int(height * 0.68)
    shoulder_dx = int(width * 0.20)   # 肩稍宽一点,接近自然头肩比

    keypoints = {
        0: (cx, y_nose),                          # nose
        1: (cx, y_neck),                          # neck
        2: (cx + shoulder_dx, y_shoulder),        # right shoulder
        5: (cx - shoulder_dx, y_shoulder),        # left shoulder
    }

    kp_colors = {
        0: (255, 0,   85),
        1: (255, 0,   0),
        2: (255, 85,  0),
        5: (170, 255, 0),
    }

    limbs = [
        (0, 1, (0, 0,   255)),   # nose → neck
        (1, 2, (255, 85, 0)),    # neck → R shoulder
        (1, 5, (170, 255, 0)),   # neck → L shoulder
    ]

    for a, b, color in limbs:
        draw.line([keypoints[a], keypoints[b]], fill=color, width=4)
    for idx, (x, y) in keypoints.items():
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=kp_colors[idx])

    return img


def _get_apose_skeleton() -> Image.Image:
    """返回 A-pose 骨架图,带磁盘缓存。"""
    if SKELETON_CACHE.exists():
        return Image.open(SKELETON_CACHE).convert("RGB")
    SKELETON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    img = _build_apose_skeleton()
    img.save(SKELETON_CACHE)
    print(f"[Γ-real] A-pose 骨架已生成并缓存: {SKELETON_CACHE}")
    return img


# ---------- call_sd ----------

def call_sd(prompt: str, negative: str = "",
            control_image: Optional[Image.Image] = None,
            w: int = SD_CHAR_W, h: int = SD_CHAR_H) -> Image.Image:
    """SD1.5 + ControlNet OpenPose。

    自动判定:
      - 输入尺寸 == 角色尺寸 → ControlNet 强度 1.0,用 A-pose 骨架
      - 其他 (场景背景板)    → ControlNet 强度 0.0,等价于禁用 ControlNet
    """
    pipe = _get_sd_pipe()

    is_character = (w, h) == (SD_CHAR_W, SD_CHAR_H)
    if is_character and control_image is None:
        control_image = _get_apose_skeleton()
    elif control_image is None:
        # 场景: 喂全黑图,ControlNet 强度设 0,等价禁用
        control_image = Image.new("RGB", (w, h), (0, 0, 0))

    if control_image.size != (w, h):
        control_image = control_image.resize((w, h), Image.LANCZOS)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        image=control_image,
        width=w,
        height=h,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0 if is_character else 0.0,
    ).images[0]
    return out


# ---------- call_mv_adapter ----------

def _split_grid(grid: Image.Image, n: int, tile_size: int = MV_TILE_SIZE) -> list[Image.Image]:
    """把 MV-Adapter 输出的横向 grid 拼图切成 n 张独立 tile。"""
    if grid.width != n * tile_size or grid.height != tile_size:
        # 容错: 万一尺寸不是预期的 768
        actual_tile_w = grid.width // n
        actual_tile_h = grid.height
        return [
            grid.crop((i * actual_tile_w, 0, (i + 1) * actual_tile_w, actual_tile_h))
            for i in range(n)
        ]
    return [
        grid.crop((i * tile_size, 0, (i + 1) * tile_size, tile_size))
        for i in range(n)
    ]


def call_mv_adapter(ref: Image.Image, prompt: str = "") -> list[Image.Image]:
    """用 subprocess 调 MV-Adapter SDXL i2mv,返回 6 张视角图。

    顺序对应 MV_ADAPTER_AZIMUTHS = [0, 45, 90, 180, 270, 315]。

    显存管理: 跑 subprocess 之前把主进程的 SD pipeline 搬到 CPU,
    让出 ~6GB 显存给 MV-Adapter SDXL 加载。跑完搬回 CUDA。
    """
    assert MV_ADAPTER_REPO.exists(), f"MV-Adapter repo 不存在: {MV_ADAPTER_REPO}"

    # 让位: 把 SD pipeline 从 GPU 搬到 CPU,清空缓存
    global _sd_pipe
    sd_was_on_cuda = False
    if _sd_pipe is not None:
        try:
            sd_was_on_cuda = next(_sd_pipe.unet.parameters()).device.type == "cuda"
        except (StopIteration, AttributeError):
            sd_was_on_cuda = False
        if sd_was_on_cuda:
            print("[Γ-real] 把 SD pipeline 搬到 CPU 让出显存给 MV-Adapter...")
            _sd_pipe.to("cpu")
            torch.cuda.empty_cache()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "ref.png"
            out_path = tmp_path / "mv_grid.png"
            ref.save(in_path)

            cmd = [
                MV_ADAPTER_PYTHON,
                "-m", "scripts.inference_i2mv_sdxl",
                "--image", str(in_path),
                "--text", prompt or "a character, full body, neutral pose",
                "--output", str(out_path),
                # 不加 --remove_bg: ref 已经是白底了, 也避开 BiRefNet dtype bug
            ]
            print(f"[Γ-real] MV-Adapter subprocess: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=MV_ADAPTER_REPO,
                capture_output=True,
                text=True,
                env={**os.environ},  # 继承 PYTHONPATH (nvdiffrast stub), HF_ENDPOINT 等
            )
            if result.returncode != 0:
                print("[Γ-real] MV-Adapter STDERR (tail):")
                print(result.stderr[-2000:])
                raise RuntimeError(
                    f"MV-Adapter subprocess failed (exit {result.returncode})"
                )

            if not out_path.exists():
                raise RuntimeError(
                    f"MV-Adapter 跑完但输出不存在: {out_path}\n"
                    f"STDOUT tail:\n{result.stdout[-1000:]}"
                )

            grid = Image.open(out_path).convert("RGB")
            tiles = _split_grid(grid, n=6, tile_size=MV_TILE_SIZE)
    finally:
        # 不管成功失败,都把 SD pipeline 搬回 GPU,等下一个角色用
        if sd_was_on_cuda and _sd_pipe is not None:
            print("[Γ-real] 把 SD pipeline 搬回 GPU...")
            _sd_pipe.to("cuda")
            torch.cuda.empty_cache()

    return tiles


# ---------- call_hunyuan3d (仍 stub) ----------

def call_hunyuan3d(ref: Image.Image) -> list[Image.Image]:
    """简版双人对话不需要斜背视角,返回空。

    cinematographer 已被改成不再调这个函数,这里只为接口兼容。
    """
    return []
