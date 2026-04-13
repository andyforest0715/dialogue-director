"""
Script Director (χ) — Dialogue Director 第一个智能体

对应论文 3.2 节，两阶段：
  χ_0: 从剧本/散文 S 抽取 {c_i}, {p_j}, {d_l}
  χ_1: 以 χ_0 输出为锚点，回到原文为每个角色和场景补全视觉描述

输入支持两种格式：
  - 剧本格式（带 INT./EXT. 场次标题 + 角色名/台词）
  - 散文故事（如《小王子》中的对话章节）
LLM 自己识别并归一化。
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# ---------- 路径配置(在这里改) ----------
INPUT_SCRIPT_PATH = "data/sunset.txt"
OUTPUT_DIR = "outputs"
DEEPSEEK_API_KEY = ""  # 在这里填你的 DeepSeek key


# ---------- 数据结构 ----------

class Character(BaseModel):
    name: str
    description: str = ""  # χ_1 补全:外貌、服装、气质等视觉细节

class Place(BaseModel):
    name: str
    description: str = ""       # χ_1 补全:布局、光照、陈设、氛围
    cast: list[str] = []        # χ_2 补全:该场景全部在场角色(说话的 + 沉默的)

class Dialogue(BaseModel):
    speaker: str   # 角色名(对应 Character.name),不用 index 方便后面按名查图
    text: str
    scene: str     # 场景名(对应 Place.name)

class ScriptStage0(BaseModel):
    """χ_0 的输出:只有名字,没有视觉描述"""
    characters: list[Character]
    places: list[Place]
    dialogues: list[Dialogue]

class Script(BaseModel):
    """χ_1 完成后的最终结构,即 T-RAG 知识库"""
    characters: list[Character]
    places: list[Place]
    dialogues: list[Dialogue]


# ---------- 名字归一化 ----------

# T-RAG 层:轻清洗。en-dash / em-dash → ASCII -, 连续 - 合并, 压缩空格
# 目的:让 T-RAG 的 key 是纯 ASCII,但保留可读结构(INT./EXT. 等)
_DASH_RE = re.compile(r"[\u2013\u2014\u2212]")  # – — −
_DASH_RUN_RE = re.compile(r"-{2,}")              # 连续 2+ 横线
_WS_RE = re.compile(r"\s+")

def normalize_name(name: str) -> str:
    """T-RAG 层的轻清洗:用于 character/place 的 name 字段。

    保留大小写和可读符号,只把非 ASCII 破折号换成 -, 连续 - 折叠成 --,
    压缩多余空格。
    """
    s = unicodedata.normalize("NFKC", name)
    s = _DASH_RE.sub("-", s)
    s = _DASH_RUN_RE.sub("--", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# 文件系统层:slug。给 vrag/ 和 storyboards/ 的目录命名用
_SLUG_INVALID_RE = re.compile(r"[^\w\u4e00-\u9fff]+")  # 保留字母数字下划线和汉字

def slugify(name: str) -> str:
    """把任意 character/place 名字转成文件系统安全的小写 slug。

    例: 'INT.BOOKSTORE--DAY' -> 'int_bookstore_day'
        'The Little Prince' -> 'the_little_prince'
        '小王子'              -> '小王子'
    """
    s = unicodedata.normalize("NFKC", name).lower().strip()
    s = _SLUG_INVALID_RE.sub("_", s)
    return s.strip("_")


# ---------- LLM 客户端 ----------

class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = "deepseek-chat"

    def chat_json(self, system: str, user: str) -> dict:
        """调用 DeepSeek 并强制返回 JSON 对象"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)


# ---------- Prompts ----------

I_0 = """你是一个专业的剧本分析师,负责从输入文本中抽取叙事元素。

输入文本可能是两种格式之一:
  (A) 标准剧本格式: 含 INT./EXT. 场次标题、全大写角色名、台词
  (B) 散文小说格式: 对话嵌在叙述段落中,通过引号和"X said"标识说话人

无论哪种格式,你都要按"先整体后局部"的链式思维完成以下三步:

第 1 步 — 通读全文,列出所有出场角色(去重),只保留 name 字段。
第 2 步 — 列出所有不同的场景/地点(去重),只保留 name 字段。
   * 剧本格式: 直接用 INT./EXT. 标题
   * 散文格式: 从叙述中提炼,例如"under the apple tree" → "Apple Tree Field"
   * 如果全篇只有一个隐含场景,就给它起一个合适的名字
第 3 步 — 按出现顺序列出每一段对话,每段包含: speaker(角色名)、text(台词原文)、scene(所属场景名)。
   * 散文里的引号内文字就是 text,叙述部分不要混进来
   * 遇到 "he said" 这种代词,要根据上下文还原成具体角色名

严格只输出 JSON,结构如下:
{
  "characters": [{"name": "..."}],
  "places": [{"name": "..."}],
  "dialogues": [{"speaker": "...", "text": "...", "scene": "..."}]
}

注意:此阶段不要补全任何视觉描述,description 字段留空或不写。
"""

I_1 = """你是一个剧本视觉化顾问。我会给你:
  1. 一段完整的原始剧本/故事文本
  2. 一份待精炼的角色名列表
  3. 一份待精炼的场景名列表

你的任务是为每个角色和场景写一段适合 Stable Diffusion 的视觉描述(description)。
**后续流程会分别生成角色和场景,然后通过抠图合成**,所以两类描述必须严格隔离 ——
角色描述里不能出现任何场景信息,场景描述里不能出现任何人物。

==========================
【角色描述规则】
==========================
长度: 25-50 个英文词。
只写**物理外观**:年龄段、体型、发色发型、肤色、面部特征、穿着、配饰。

严禁出现以下词类(违反会破坏后续多视角生成):
  ✗ 姿态/动作: standing, sitting, walking, holding, looking, leaning
  ✗ 位置/环境: indoors, outdoors, in the room, at the bar, on the street, behind a desk
  ✗ 情绪/性格: friendly, relaxed, thoughtful, charming, sad, confident, warm
  ✗ 关系: with celine, next to jesse, talking to

每个角色都是孤立的纯人物描述,想象他在白色摄影棚里拍角色设定表。

例子:
  ✓ "man in his mid-30s, lean build, short dark brown hair, light stubble, white t-shirt, faded blue jeans, brown leather wristband"
  ✗ "young man in casual clothing, short hair, standing indoors"     ← 含 pose+location
  ✗ "relaxed friendly man with thoughtful expression"                ← 含情绪
  ✗ "Jesse standing next to Celine in the bookstore"                 ← 含关系+location

==========================
【场景描述规则】
==========================
长度: 25-50 个英文词。
只写**空间本身**:布局、家具陈设、光照、时间、天气、色调、氛围。
描述这个场景**没有任何人在场**时的样子(因为后续会单独抠图贴角色进去)。

严禁出现以下:
  ✗ 任何人物: people, customers, jesse, celine, a person, figures, crowd
  ✗ 任何由人物引发的动作或事件

例子:
  ✓ "interior of a small independent bookstore, tall wooden shelves filled with books, warm afternoon sunlight through large front windows, hardwood floor, a few reading lamps"
  ✗ "bookstore where Jesse and Celine first meet"   ← 提到了人物
  ✗ "garden where they walk and talk"               ← 含人物+动作

==========================
【输出格式】
==========================
严格只输出 JSON,name 必须与输入列表的字符串**完全一致**(逐字符匹配):
{
  "characters": [{"name": "...", "description": "..."}],
  "places": [{"name": "...", "description": "..."}]
}
"""

I_2 = """你是一个剧本场记。我会给你:
  1. 一段完整的原始剧本/故事文本
  2. 一份场景列表
  3. 一份角色列表
  4. 每个场景里**已经说过话**的角色名单(确定性预计算的结果)

你的任务:对每个场景,找出**在场但没有说话**的角色 —— 即出现在该场景的叙述、
动作或他人对白提及中,但本身没有台词的角色。这些"沉默在场角色"对后续画分镜
(尤其是过肩镜头和群像)很重要。

【判断规则】
1. 只算**人物性**角色,且必须能在画面里出现
2. 已经在 speaking_cast 里的角色不要重复列出
3. 如果一个场景里没有任何沉默在场角色,返回空列表 []
4. 不要凭空编造角色,只能从给定的角色列表中选

【输出格式】严格只输出 JSON,places 的 name 必须与输入完全一致:
{
  "places": [
    {"name": "<场景名>", "silent_cast": ["<角色名>", ...]}
  ]
}
"""


# ---------- 两阶段函数 ----------

def chi_0(script_text: str, llm: DeepSeekClient) -> ScriptStage0:
    """χ_0: 实体识别 — 抽出角色、场景、对话三类元素"""
    raw = llm.chat_json(system=I_0, user=script_text)
    stage0 = ScriptStage0.model_validate(raw)

    # 归一化所有 name 字段(en-dash → ASCII, 压缩空格),并保持 dialogues 的引用一致
    char_map = {c.name: normalize_name(c.name) for c in stage0.characters}
    place_map = {p.name: normalize_name(p.name) for p in stage0.places}

    return ScriptStage0(
        characters=[Character(name=char_map[c.name]) for c in stage0.characters],
        places=[Place(name=place_map[p.name]) for p in stage0.places],
        dialogues=[
            Dialogue(
                speaker=char_map.get(d.speaker, normalize_name(d.speaker)),
                text=d.text,
                scene=place_map.get(d.scene, normalize_name(d.scene)),
            )
            for d in stage0.dialogues
        ],
    )


def chi_1(script_text: str, stage0: ScriptStage0, llm: DeepSeekClient) -> Script:
    """χ_1: 视觉细节精炼 — 为每个角色和场景补全 description"""
    user_msg = (
        f"=== 原始文本 ===\n{script_text}\n\n"
        f"=== 待精炼角色列表 ===\n"
        f"{json.dumps([c.name for c in stage0.characters], ensure_ascii=False)}\n\n"
        f"=== 待精炼场景列表 ===\n"
        f"{json.dumps([p.name for p in stage0.places], ensure_ascii=False)}"
    )
    raw = llm.chat_json(system=I_1, user=user_msg)

    # 把 χ_1 补全的 description 合并回 stage0,保持 dialogues 不变
    refined_chars = {c["name"]: c["description"] for c in raw["characters"]}
    refined_places = {p["name"]: p["description"] for p in raw["places"]}

    return Script(
        characters=[
            Character(name=c.name, description=refined_chars.get(c.name, ""))
            for c in stage0.characters
        ],
        places=[
            Place(name=p.name, description=refined_places.get(p.name, ""))
            for p in stage0.places
        ],
        dialogues=stage0.dialogues,
    )


def chi_2(script_text: str, refined: Script, llm: DeepSeekClient) -> Script:
    """χ_2: 为每个场景补全 cast 字段(在场角色 = 说话的 + 沉默在场的)。

    说话的部分确定性算出,沉默的部分调 LLM 抽取,二者 union。
    """
    # 1. 确定性:从 dialogues groupby scene 得到说话角色
    speaking_cast: dict[str, list[str]] = {p.name: [] for p in refined.places}
    for d in refined.dialogues:
        if d.scene in speaking_cast and d.speaker not in speaking_cast[d.scene]:
            speaking_cast[d.scene].append(d.speaker)

    # 2. LLM:抽出每个场景的沉默在场角色
    user_msg = (
        f"=== 原始文本 ===\n{script_text}\n\n"
        f"=== 场景列表 ===\n"
        f"{json.dumps([p.name for p in refined.places], ensure_ascii=False)}\n\n"
        f"=== 角色列表 ===\n"
        f"{json.dumps([c.name for c in refined.characters], ensure_ascii=False)}\n\n"
        f"=== 各场景已说话角色(speaking_cast) ===\n"
        f"{json.dumps(speaking_cast, ensure_ascii=False, indent=2)}"
    )
    raw = llm.chat_json(system=I_2, user=user_msg)
    silent_map = {p["name"]: p.get("silent_cast", []) for p in raw["places"]}

    # 3. union 成最终 cast,过滤掉幻觉(不在角色表里的)
    valid_chars = {c.name for c in refined.characters}
    new_places = []
    for p in refined.places:
        speakers = speaking_cast.get(p.name, [])
        silents = [s for s in silent_map.get(p.name, []) if s in valid_chars and s not in speakers]
        full_cast = speakers + silents
        new_places.append(Place(name=p.name, description=p.description, cast=full_cast))

    return Script(
        characters=refined.characters,
        places=new_places,
        dialogues=refined.dialogues,
    )


def run_script_director(script_text: str, llm: Optional[DeepSeekClient] = None) -> Script:
    """完整 χ 流程:χ_0 → χ_1 → χ_2,返回 T-RAG 知识库"""
    llm = llm or DeepSeekClient()
    stage0 = chi_0(script_text, llm)
    print(f"[χ_0] 抽出 {len(stage0.characters)} 个角色, "
          f"{len(stage0.places)} 个场景, {len(stage0.dialogues)} 段对话")
    refined = chi_1(script_text, stage0, llm)
    print(f"[χ_1] 已为所有角色和场景补全视觉描述")
    final = chi_2(script_text, refined, llm)
    print(f"[χ_2] 已为所有场景补全在场角色名单")
    return final


# ---------- 持久化(T-RAG 知识库,按名字精确查) ----------

def project_dir(script_path: str | Path, output_base: str | Path = OUTPUT_DIR) -> Path:
    """从输入剧本路径派生该项目的输出子目录。

    例: 'data/sample.txt' + 'outputs' -> Path('outputs/sample')
    下游 Γ / ζ 也调用这个函数,保证三个 agent 写到同一个目录。
    """
    stem = Path(script_path).stem
    return Path(output_base) / stem


def save_trag(script: Script, out_dir: str | Path):
    """落盘成 JSON。后续摄影师和分镜师按 name 字典查询。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trag.json").write_text(
        script.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"[T-RAG] 已写入 {out_dir / 'trag.json'}")


def load_trag(out_dir: str | Path) -> Script:
    return Script.model_validate_json(
        (Path(out_dir) / "trag.json").read_text(encoding="utf-8")
    )


if __name__ == "__main__":
    text = Path(INPUT_SCRIPT_PATH).read_text(encoding="utf-8")
    script = run_script_director(text)
    out = project_dir(INPUT_SCRIPT_PATH, OUTPUT_DIR)
    save_trag(script, out)
    print("\n--- 角色 ---")
    for c in script.characters:
        print(f"  {c.name}: {c.description[:80]}...")
    print("\n--- 场景 ---")
    for p in script.places:
        print(f"  {p.name}: {p.description[:80]}...")
        print(f"    cast: {p.cast}")
    print(f"\n--- 对话(前 3 条) ---")
    for d in script.dialogues[:3]:
        print(f"  [{d.scene}] {d.speaker}: {d.text}")
