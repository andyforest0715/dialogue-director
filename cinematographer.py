"""
Cinematographer (Γ) — Dialogue Director 第二个智能体

对应论文 3.3 节,两阶段:
  Γ_0: 文本描述 → SD1.5 → ref.png / plate.png
       角色另加 ControlNet OpenPose A-pose 骨架,锁正面、画面占比、头部位置
  Γ_1: ref.png → MV-Adapter (6 视角) + Hunyuan3D-1 (2 斜背) → 8 视角肖像

输出: outputs/<script>/vrag/
        characters/<slug>/view_<idx>_<slug>.png + ..._rgba.png + meta.json
        places/<slug>/plate.png + meta.json

第一版策略: 扩散模型全部 stub(灰色占位图),rembg 和 meta 计算真做。
真模型在 diffusion_backend.py 里实现,通过 DD_BACKEND 环境变量切换:
  DD_BACKEND=stub  (默认) → diffusion_backend_stub
  DD_BACKEND=real         → diffusion_backend
其余代码不动。
"""

import json
import os
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from script_director import (
    Script, load_trag, project_dir, slugify,
    INPUT_SCRIPT_PATH, OUTPUT_DIR,
)

# ---------- Backend 开关 ----------
# DD_BACKEND=stub (默认): 用 PIL 占位图,不需要 GPU,本地能跑
# DD_BACKEND=real:        用真扩散模型(SD1.5 + MV-Adapter + Hunyuan3D-1 stage1),需要 CUDA
BACKEND = os.environ.get("DD_BACKEND", "stub").lower()
if BACKEND == "real":
    from diffusion_backend import call_sd, call_mv_adapter, call_hunyuan3d
else:
    from diffusion_backend_stub import call_sd, call_mv_adapter, call_hunyuan3d
print(f"[Γ] backend = {BACKEND}")


# ---------- 配置 ----------

# 角色参考图尺寸 (头肩证件照构图,跟 diffusion_backend.SD_CHAR_W/H 对齐)
CHAR_IMG_W, CHAR_IMG_H = 768, 768
# 场景背景板 16:9 横构图
PLACE_IMG_W, PLACE_IMG_H = 1024, 576

# 论文 3.3 节的 θ_base — 全局风格基线提示词
THETA_BASE_CHAR = (
    "headshot portrait, head and shoulders only, front view, looking at camera, "
    "neutral expression, plain white background, soft studio lighting, "
    "passport photo style, centered, sharp focus, photorealistic, high detail"
)
THETA_BASE_PLACE = (
    "empty location, no people, cinematic photograph, natural lighting, high detail"
)

NEGATIVE_CHAR = (
    "full body, torso, waist, hips, legs, arms, hands, fingers, "
    "sitting, action, dynamic pose, cropped face, multiple people, "
    "low quality, blurry, deformed, text, watermark"
)
NEGATIVE_PLACE = (
    "people, person, figure, crowd, character, human, text, watermark"
)

# 6 视角定义,全部来自 MV-Adapter SDXL i2mv (azimuth 顺序见 diffusion_backend)
# slug 命名原则: 用语义而非角度,方便 ζ 阶段按"我要什么角度"思考
# azimuth 0° = 正面,顺时针递增
VIEWS: list[dict] = [
    {"idx": 0, "slug": "front",         "source": "mv_adapter", "azimuth": 0},
    {"idx": 1, "slug": "front_right",   "source": "mv_adapter", "azimuth": 45},
    {"idx": 2, "slug": "right_profile", "source": "mv_adapter", "azimuth": 90},
    {"idx": 3, "slug": "back",          "source": "mv_adapter", "azimuth": 180},
    {"idx": 4, "slug": "left_profile",  "source": "mv_adapter", "azimuth": 270},
    {"idx": 5, "slug": "front_left",    "source": "mv_adapter", "azimuth": 315},
]


# ---------- meta.json schema ----------

class ViewMeta(BaseModel):
    idx: int
    slug: str
    source: str
    azimuth: int
    file: str             # 相对路径 (相对于角色目录)
    rgba_file: str
    bbox: tuple[int, int, int, int]   # x, y, w, h (在 RGBA 图里)
    char_height_px: int   # = bbox[3], 给 ζ 做确定性缩放用


class CharacterAssets(BaseModel):
    name: str
    slug: str
    canvas_size: tuple[int, int]
    views: dict[str, ViewMeta]   # 以 view slug 为 key


class PlaceAssets(BaseModel):
    name: str
    slug: str
    canvas_size: tuple[int, int]
    plate_file: str


# ---------- 真实现: 抠图 + bbox ----------

def _matte(rgb_image: Image.Image) -> Image.Image:
    """rembg 抠图,返回 RGBA。

    需要 `pip install rembg`。未安装时 fallback 成"白底变透明",
    够用于 stub 阶段验证 pipeline。
    """
    try:
        from rembg import remove
        return remove(rgb_image)
    except ImportError:
        rgba = rgb_image.convert("RGBA")
        pixels = rgba.load()
        for y in range(rgba.height):
            for x in range(rgba.width):
                r, g, b, _ = pixels[x, y]
                if r > 235 and g > 235 and b > 235:
                    pixels[x, y] = (255, 255, 255, 0)
        return rgba


def _compute_bbox(rgba: Image.Image) -> tuple[int, int, int, int]:
    """从 RGBA 的 alpha 通道扫出非透明区域 bbox = (x, y, w, h)。"""
    alpha = rgba.split()[-1]
    box = alpha.getbbox()
    if box is None:
        return (0, 0, rgba.width, rgba.height)
    left, upper, right, lower = box
    return (left, upper, right - left, lower - upper)


# ---------- 单角色 / 单场景生成 ----------

def generate_character(name: str, description: str, vrag_dir: Path) -> CharacterAssets:
    """对单个角色跑 Γ_0 + Γ_1 + 抠图 + meta。"""
    slug = slugify(name)
    out_dir = vrag_dir / "characters" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Γ_0: 文生图 (real backend 自动加 A-pose ControlNet 骨架)
    prompt = f"{description}, {THETA_BASE_CHAR}"
    ref = call_sd(prompt, negative=NEGATIVE_CHAR, control_image=None)

    # Γ_1: 多视角扩散 (real backend = MV-Adapter SDXL i2mv subprocess)
    # 注意 prompt 也传过去,让 MV-Adapter 知道角色描述
    mv_views = call_mv_adapter(ref, prompt=prompt)
    assert len(mv_views) == len(VIEWS), (
        f"call_mv_adapter 返回 {len(mv_views)} 张, 期望 {len(VIEWS)} 张"
    )
    view_imgs: dict[int, Image.Image] = {v["idx"]: mv_views[i] for i, v in enumerate(VIEWS)}

    # 落盘 + 抠图 + 算 bbox
    views_meta: dict[str, ViewMeta] = {}
    for v in VIEWS:
        img = view_imgs[v["idx"]]
        rgb_name = f"view_{v['idx']}_{v['slug']}.png"
        rgba_name = f"view_{v['idx']}_{v['slug']}_rgba.png"
        img.save(out_dir / rgb_name)
        rgba = _matte(img)
        rgba.save(out_dir / rgba_name)
        bbox = _compute_bbox(rgba)
        views_meta[v["slug"]] = ViewMeta(
            idx=v["idx"], slug=v["slug"], source=v["source"], azimuth=v["azimuth"],
            file=rgb_name, rgba_file=rgba_name,
            bbox=bbox, char_height_px=bbox[3],
        )

    assets = CharacterAssets(
        name=name, slug=slug,
        canvas_size=(CHAR_IMG_W, CHAR_IMG_H),
        views=views_meta,
    )
    (out_dir / "meta.json").write_text(assets.model_dump_json(indent=2), encoding="utf-8")
    return assets


def generate_place(name: str, description: str, vrag_dir: Path) -> PlaceAssets:
    """对单个场景跑 Γ_0,出空场景背景板。"""
    slug = slugify(name)
    out_dir = vrag_dir / "places" / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = f"{description}, {THETA_BASE_PLACE}"
    plate = call_sd(prompt, negative=NEGATIVE_PLACE, w=PLACE_IMG_W, h=PLACE_IMG_H)
    plate.save(out_dir / "plate.png")

    assets = PlaceAssets(
        name=name, slug=slug,
        canvas_size=(PLACE_IMG_W, PLACE_IMG_H),
        plate_file="plate.png",
    )
    (out_dir / "meta.json").write_text(assets.model_dump_json(indent=2), encoding="utf-8")
    return assets


# ---------- 主流程 ----------

def run_cinematographer(script: Script, project_root: Path) -> dict:
    """跑完整 Γ: 给 T-RAG 里的所有 character 和 place 生成 V-RAG。"""
    vrag_dir = project_root / "vrag"
    vrag_dir.mkdir(parents=True, exist_ok=True)

    chars: dict[str, CharacterAssets] = {}
    for c in script.characters:
        print(f"[Γ] character: {c.name}")
        chars[c.name] = generate_character(c.name, c.description, vrag_dir)

    places: dict[str, PlaceAssets] = {}
    for p in script.places:
        print(f"[Γ] place: {p.name}")
        places[p.name] = generate_place(p.name, p.description, vrag_dir)

    return {"characters": chars, "places": places}


if __name__ == "__main__":
    proj = project_dir(INPUT_SCRIPT_PATH, OUTPUT_DIR)
    script = load_trag(proj)
    print(f"[Γ] 加载 T-RAG: {proj / 'trag.json'}")
    print(f"    {len(script.characters)} 角色, {len(script.places)} 场景")
    run_cinematographer(script, proj)
    print(f"\n[Γ] 完成。V-RAG 产物在: {proj / 'vrag'}")
