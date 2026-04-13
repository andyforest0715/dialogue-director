"""
Storyboard Maker (ζ) — Dialogue Director 第三个智能体

对应论文 3.4 节,三阶段(简版双人对话):
  ζ_0: 视角选择 — 双人对话简版用确定性规则,无需 LLM
  ζ_1: 布局规划 — 5 个 shot 模板的位置/缩放/层级在 SHOTS 常量里硬编码
  ζ_2: 图像合成 — PIL paste, 缩放按 V-RAG meta 里的 char_height_px 确定性归一化

输入: outputs/<script>/{trag.json, vrag/}
输出: outputs/<script>/storyboards/
        scene_<idx>_<slug>/
            shot_<idx>_<name>.png   ← 5 张 shot 模板/scene
            shots.json
        timeline.json               ← dialogue idx → (scene, shot) 派发
"""

import json
from pathlib import Path
from typing import Optional

from PIL import Image
from pydantic import BaseModel

from script_director import (
    Script, Dialogue, load_trag, project_dir, slugify,
    INPUT_SCRIPT_PATH, OUTPUT_DIR,
)
from cinematographer import CharacterAssets, PlaceAssets


# ---------- 配置 ----------

SHOT_W, SHOT_H = 1280, 720   # 16:9 720p


# 5 个 shot 模板。每个 shot 描述了用哪两个 view、贴在哪、多大、谁在前面。
#   pos:   角色 bbox **底边** 在画布上的相对坐标 (x, y in 0-1)
#          x = 角色横向中心位置, y = 底边贴到画面这个位置 (1.0 = 画面最底)
#   scale: 角色 bbox 高度占画布高度的比例
#          头肩近景的角色比半身更紧凑,scale 取 0.95-1.15 让角色充满画面下半部
#   z:     贴图层级,越大越靠前(前景)
#
# 注:shot 0 和 shot 3 视角组合相同(都是 front_right + front_left),
#    差异通过 scale 体现 — shot 0 较远的双人镜头, shot 3 更近
SHOTS: list[dict] = [
    {
        "idx": 0,
        "name": "two_shot_profile",   # 双人侧面面对面,建立镜头
        # 注:经实测 MV-Adapter 的 front_right(az=45) 实际脸朝左,front_left(az=315) 实际脸朝右
        # 所以画面左的角色用 front_left (才能朝右看对方),画面右用 front_right
        "A": {"view": "front_left",  "pos": (0.30, 1.00), "scale": 0.95, "z": 1},
        "B": {"view": "front_right", "pos": (0.70, 1.00), "scale": 0.95, "z": 1},
    },
    {
        "idx": 1,
        "name": "A_front_B_back_fg",  # A 正面在 1/2 处, B 背影占左前景
        "A": {"view": "front", "pos": (0.62, 1.00), "scale": 1.00, "z": 1},
        "B": {"view": "back",  "pos": (0.18, 1.05), "scale": 1.20, "z": 2},
    },
    {
        "idx": 2,
        "name": "B_front_A_back_fg",  # shot 1 镜像
        "A": {"view": "back",  "pos": (0.18, 1.05), "scale": 1.20, "z": 2},
        "B": {"view": "front", "pos": (0.62, 1.00), "scale": 1.00, "z": 1},
    },
    {
        "idx": 3,
        "name": "45deg",              # 双人 45 度,更近
        "A": {"view": "front_left",  "pos": (0.32, 1.02), "scale": 1.10, "z": 1},
        "B": {"view": "front_right", "pos": (0.68, 1.02), "scale": 1.10, "z": 1},
    },
    {
        "idx": 4,
        "name": "45deg_mirror",
        "A": {"view": "front_right", "pos": (0.32, 1.02), "scale": 1.10, "z": 1},
        "B": {"view": "front_left",  "pos": (0.68, 1.02), "scale": 1.10, "z": 1},
    },
]


# ---------- 数据结构 ----------

class ShotMeta(BaseModel):
    idx: int
    name: str
    file: str           # 相对于 scene 目录
    A_char: str
    A_view: str
    B_char: str
    B_view: str


class SceneShots(BaseModel):
    scene_name: str
    scene_slug: str
    A: str              # 角色名
    B: str
    shots: list[ShotMeta]


class TimelineEntry(BaseModel):
    dialogue_idx: int
    scene_slug: str
    scene_name: str
    shot_idx: int
    shot_name: str
    speaker: str
    text: str


# ---------- 加载器 ----------

def load_character_assets(vrag_dir: Path, char_name: str) -> CharacterAssets:
    p = vrag_dir / "characters" / slugify(char_name) / "meta.json"
    return CharacterAssets.model_validate_json(p.read_text(encoding="utf-8"))


def load_place_assets(vrag_dir: Path, place_name: str) -> PlaceAssets:
    p = vrag_dir / "places" / slugify(place_name) / "meta.json"
    return PlaceAssets.model_validate_json(p.read_text(encoding="utf-8"))


# ---------- A/B 角色确定 ----------

def determine_AB(scene_name: str, dialogues: list[Dialogue]) -> Optional[tuple[str, str]]:
    """以 scene 内出现顺序的前两位说话角色为 A、B。不到 2 人返回 None。"""
    seen: list[str] = []
    for d in dialogues:
        if d.scene == scene_name and d.speaker not in seen:
            seen.append(d.speaker)
            if len(seen) >= 2:
                return seen[0], seen[1]
    return None


# ---------- 合成器 ζ_2 ----------

def _scaled_paste(base: Image.Image, fg_rgba_path: Path,
                  view_meta_h: int,
                  pos: tuple[float, float],
                  scale: float,
                  anchor: str = "bottom") -> Image.Image:
    """把一张 RGBA 角色 cutout 按目标 scale 缩放后贴到 base 上。

    view_meta_h: V-RAG meta 里记录的 char_height_px (原图 bbox 高度)
    pos: 角色 bbox 锚点在 base 上的相对坐标 (0-1, 0-1)
    scale: 缩放后的角色 bbox 高度占 base 高度的比例
    anchor: 'bottom' = pos[1] 指 bbox 底边 y (适合头肩近景, 角色脚/胸口对齐画面底)
            'center' = pos[1] 指 bbox 中心 y (旧行为, 角色居中)
    """
    fg = Image.open(fg_rgba_path).convert("RGBA")

    # 1. 按 scale 算缩放系数,resize 整张图
    target_h = int(base.height * scale)
    factor = target_h / max(view_meta_h, 1)
    new_size = (max(1, int(fg.width * factor)), max(1, int(fg.height * factor)))
    fg = fg.resize(new_size, Image.LANCZOS)

    # 2. 找 resize 后的 bbox
    alpha = fg.split()[-1]
    bbox = alpha.getbbox() or (0, 0, fg.width, fg.height)
    cx_in_fg = (bbox[0] + bbox[2]) // 2
    if anchor == "bottom":
        anchor_y_in_fg = bbox[3]                    # bbox 底边
    else:
        anchor_y_in_fg = (bbox[1] + bbox[3]) // 2   # bbox 中心

    # 3. 平移到目标位置
    target_cx = int(base.width * pos[0])
    target_anchor_y = int(base.height * pos[1])
    paste_x = target_cx - cx_in_fg
    paste_y = target_anchor_y - anchor_y_in_fg

    base.paste(fg, (paste_x, paste_y), fg)
    return base


def compose_shot(plate_path: Path,
                 char_A: CharacterAssets, char_A_dir: Path,
                 char_B: CharacterAssets, char_B_dir: Path,
                 shot: dict) -> Image.Image:
    """给一个 shot 模板,合成一张完整画面 (背景板 + A + B)。"""
    plate = Image.open(plate_path).convert("RGBA").resize((SHOT_W, SHOT_H), Image.LANCZOS)

    # 按 z 升序贴 (小 z 先贴 = 在后面)
    layers = sorted(
        [("A", shot["A"], char_A, char_A_dir),
         ("B", shot["B"], char_B, char_B_dir)],
        key=lambda t: t[1]["z"],
    )
    for _who, spec, char, char_dir in layers:
        view = char.views[spec["view"]]
        plate = _scaled_paste(
            plate, char_dir / view.rgba_file,
            view_meta_h=view.char_height_px,
            pos=spec["pos"], scale=spec["scale"],
        )
    return plate.convert("RGB")


# ---------- 单 scene 处理 ----------

def make_scene_shots(scene_name: str, A_name: str, B_name: str,
                     vrag_dir: Path, storyboards_dir: Path,
                     scene_idx: int) -> SceneShots:
    """给一个双人 scene 生成 5 张 shot 模板和 shots.json。"""
    scene_slug = slugify(scene_name)
    scene_out = storyboards_dir / f"scene_{scene_idx:02d}_{scene_slug}"
    scene_out.mkdir(parents=True, exist_ok=True)

    char_A = load_character_assets(vrag_dir, A_name)
    char_B = load_character_assets(vrag_dir, B_name)
    place = load_place_assets(vrag_dir, scene_name)

    char_A_dir = vrag_dir / "characters" / char_A.slug
    char_B_dir = vrag_dir / "characters" / char_B.slug
    plate_path = vrag_dir / "places" / place.slug / place.plate_file

    shot_metas: list[ShotMeta] = []
    for shot in SHOTS:
        img = compose_shot(plate_path, char_A, char_A_dir, char_B, char_B_dir, shot)
        fname = f"shot_{shot['idx']}_{shot['name']}.png"
        img.save(scene_out / fname)
        shot_metas.append(ShotMeta(
            idx=shot["idx"], name=shot["name"], file=fname,
            A_char=A_name, A_view=shot["A"]["view"],
            B_char=B_name, B_view=shot["B"]["view"],
        ))

    assets = SceneShots(
        scene_name=scene_name, scene_slug=scene_slug,
        A=A_name, B=B_name, shots=shot_metas,
    )
    (scene_out / "shots.json").write_text(
        assets.model_dump_json(indent=2), encoding="utf-8"
    )
    return assets


# ---------- Timeline 派发 (ζ_0 简版规则) ----------

def make_timeline(script: Script, scene_assets: dict[str, SceneShots]) -> list[TimelineEntry]:
    """按确定性规则给每段对话派发 shot。

    规则:
      - 每个 scene 的第 1 句  → shot 0 (建立镜头)
      - 后续 A 说话           → shot 1 (A 正面 + B 背影)
      - 后续 B 说话           → shot 2 (B 正面 + A 背影)
      - shot 3, 4 (45 度)     不在 timeline 里使用,留作备选模板/未来用 LLM 选
    """
    timeline: list[TimelineEntry] = []
    seen_scenes: set[str] = set()

    for i, d in enumerate(script.dialogues):
        if d.scene not in scene_assets:
            print(f"[ζ] WARN: dialogue {i} 的 scene '{d.scene}' 没有 shot pool, 跳过")
            continue
        sa = scene_assets[d.scene]
        is_first = d.scene not in seen_scenes
        seen_scenes.add(d.scene)

        if is_first:
            shot_idx, shot_name = 0, "two_shot_profile"
        elif d.speaker == sa.A:
            shot_idx, shot_name = 1, "A_front_B_back_fg"
        elif d.speaker == sa.B:
            shot_idx, shot_name = 2, "B_front_A_back_fg"
        else:
            # 第三人插话(简版用不到),退化到 two_shot
            shot_idx, shot_name = 0, "two_shot_profile"

        timeline.append(TimelineEntry(
            dialogue_idx=i,
            scene_slug=sa.scene_slug, scene_name=d.scene,
            shot_idx=shot_idx, shot_name=shot_name,
            speaker=d.speaker, text=d.text,
        ))
    return timeline


# ---------- 主流程 ----------

def run_storyboard_maker(script: Script, project_root: Path):
    vrag_dir = project_root / "vrag"
    storyboards_dir = project_root / "storyboards"
    storyboards_dir.mkdir(parents=True, exist_ok=True)

    # 1. 给每个双人 scene 生成 5 张 shot 模板
    scene_assets: dict[str, SceneShots] = {}
    for i, place in enumerate(script.places):
        AB = determine_AB(place.name, script.dialogues)
        if AB is None:
            print(f"[ζ] WARN: scene '{place.name}' 不到 2 个说话角色, 跳过")
            continue
        A_name, B_name = AB
        print(f"[ζ] scene {i+1}: '{place.name}'  A={A_name}  B={B_name}")
        scene_assets[place.name] = make_scene_shots(
            place.name, A_name, B_name, vrag_dir, storyboards_dir, i + 1,
        )

    # 2. 派发 timeline
    timeline = make_timeline(script, scene_assets)
    (storyboards_dir / "timeline.json").write_text(
        json.dumps([e.model_dump() for e in timeline], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ζ] timeline: {len(timeline)} 条目")
    print(f"[ζ] 完成。storyboards 在: {storyboards_dir}")


if __name__ == "__main__":
    proj = project_dir(INPUT_SCRIPT_PATH, OUTPUT_DIR)
    script = load_trag(proj)
    print(f"[ζ] 加载 T-RAG: {proj / 'trag.json'}")
    run_storyboard_maker(script, proj)
