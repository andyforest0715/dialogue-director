# Dialogue Director

**Dialogue Director**官方实现，把对话剧本(或带对话的散文)自动转成多视角电影分镜画面,基于多智能体协作。


| 智能体 | 职责 | 实现 |
|---|---|---|
| **χ Script Director** | 从剧本抽取角色 / 场景 / 对话三元组,补全视觉描述 | 三阶段 LLM 调用 |
| **Γ Cinematographer** | 文本 → 角色参考图 → 多视角肖像 + 场景背景板 | SD1.5 + ControlNet OpenPose + MV-Adapter |
| **ζ Storyboard Maker** | 用 V-RAG 素材按电影构图模板合成最终分镜 | PIL + 确定性派发规则 |

## 架构

```
对话剧本 (.txt)
     │
     ▼  χ (DeepSeek)
T-RAG: trag.json  (characters/places/dialogues + 视觉描述)
     │
     ▼  Γ (SD1.5 + MV-Adapter, GPU)
V-RAG: vrag/{characters,places}/  (6 视角角色 + 场景 plate + meta.json)
     │
     ▼  ζ (PIL 合成)
storyboards/  (5 shot 模板/scene + timeline.json)
```

## 依赖

项目分两端运行:

- **本地 (无 GPU)**: 跑 χ 和 ζ,Γ 使用 stub 占位图。需要 LLM API key。
- **云端 (有 GPU)**: 跑 Γ real 模式。需要 ~24GB VRAM (RTX 3090 / 4090),CUDA 12.x 或 13.x。

### 本地

```bash
# Python 3.10+ 即可
pip install -r requirements.txt
```

### 云端 GPU

**实测可工作的版本组合** :

| 组件 | 版本 |
|---|---|
| Python | 3.12.11 |
| PyTorch | 2.9.1+cu130 |
| CUDA Toolkit | 13.0 |
| GPU | RTX 3090 24GB |
| 驱动 | 580.105.08 |

**注意**:

1. **不要 `pip install torch`** — 用云镜像预装的 PyTorch,版本对硬件已经匹配
2. **装其他依赖前先快照 torch 版本**,装完检查没被覆盖:
   ```bash
   pip freeze | grep -iE '^(torch|nvidia)' > /tmp/before.txt
   pip install -r requirements-gpu.txt
   pip freeze | grep -iE '^(torch|nvidia)' > /tmp/after.txt
   diff /tmp/before.txt /tmp/after.txt   # 应该为空
   ```
3. **PyTorch 2.x 任意小版本都能跑**,只要跟 CUDA 大版本匹配 (cu118 / cu121 / cu124 / cu128 / cu130 都行)

```bash
pip install -r requirements-gpu.txt
```

### Python 版本

| Python | 状态 |
|---|---|
| 3.10 | 应该能跑 (未测) |
| 3.11 | 应该能跑 (未测) |
| **3.12** | **实测通过** |
| 3.13 | 不建议 (部分 ML 库还没适配) |

## 快速开始

### 1. 配置

```bash
cp .env.example .env
# 编辑 .env:
#   DEEPSEEK_API_KEY=sk-...
#   DD_INPUT_SCRIPT=data/sample.txt
#   DD_OUTPUT_DIR=outputs
#   DD_BACKEND=stub      # 本地用 stub, 云端改 real
```

### 2. 准备剧本

支持两种格式,LLM 自动识别:

**(A) 标准剧本格式**

```
INT.BOOKSTORE--DAY
JESSE
You wanna get some coffee or something?
CELINE
Didn't he just say you have a plane to catch?
```

**(B) 散文格式** (对话嵌在叙述里)

```
"Good morning," said the fox.
"Good morning," the little prince responded politely.
```

放到 `data/sample.txt`。

### 3. 跑

本地 stub 模式 (验证 pipeline,出占位图):

```bash
make chi          # 出 outputs/sample/trag.json
make gamma-stub   # 出 outputs/sample/vrag/    (PIL 占位图)
make zeta         # 出 outputs/sample/storyboards/
```

完整真模型流程 (需要云端 GPU):

```bash
# 本地
make chi
make sync-up                   # rsync trag.json → 云端

# 云端 (ssh 进 GPU 实例后)
DD_BACKEND=real make gamma-real

# 本地
make sync-down                 # rsync vrag/ ← 云端
make zeta
```

## 5 个 Shot 模板

ζ 为每个双人对话场景生成 5 张 shot 模板,对应电影里的经典构图:

| Shot | 名称 | 用途 |
|---|---|---|
| 0 | two_shot_profile | 建立镜头,双人面对面,scene 第一句必用 |
| 1 | A_front_B_back_fg | A 的过肩镜头 (B 背影占左前景) |
| 2 | B_front_A_back_fg | B 的过肩镜头 (A 背影占左前景) |
| 3 | 45deg | 双人 45° 中景,情绪升温 |
| 4 | 45deg_mirror | shot 3 的反向 |

派发规则 (在 `storyboard_maker.py:make_timeline`):

- scene 第 1 句 → shot 0
- 之后 A 说话 → shot 1
- 之后 B 说话 → shot 2
- shot 3/4 暂未派发,留给未来 LLM 选 shot 升级

## 输出结构

```
outputs/sample/
├── trag.json                    # χ 产出
├── vrag/                        # Γ 产出
│   ├── characters/
│   │   └── jesse/
│   │       ├── view_0_front.png       # ref + 5 视角
│   │       ├── view_0_front_rgba.png  # rembg 抠图
│   │       └── meta.json              # bbox + char_height_px (给 ζ 缩放用)
│   └── places/
│       └── int_bookstore_day/
│           └── plate.png
└── storyboards/                 # ζ 产出
    ├── scene_01_int_bookstore_day/
    │   ├── shot_0_two_shot_profile.png
    │   ├── shot_1_A_front_B_back_fg.png
    │   └── ...
    └── timeline.json            # dialogue idx → shot 派发
```

## 关键设计决策

**为什么角色和场景分开生成再合成?**

之前踩过的坑: 端到端"角色 + 场景"一起生时,角色大小不一致,合成出来比例错乱。改成方案 b: 角色生头肩证件照 → rembg 抠图 → 记录 bbox → ζ 合成时按 `char_height_px` **确定性缩放**。从源头消除 scale 飘移。

**为什么 Γ 用 subprocess 调 MV-Adapter 而不是 import?**

MV-Adapter 是 git clone 的 repo,不是 pip 包。subprocess 隔离了它的依赖环境,避免 import 路径地狱;每次调用的冷启动开销 (~10s) 跟扩散本身 (1-3 分钟) 比起来可忽略。

**为什么 SHOTS 模板硬编码而不是 LLM 选?**

双人对话场景的电影语法收敛性极高 (90% 是 "建立镜头 → 乒乓 OTS"),硬编码规则就够。SHOTS 里的 `pos`/`scale` 是肉眼调出的视觉参数,LLM 没有眼睛,这层不该交给 LLM。

未来升级路径见 `make_timeline()` 的 docstring。

## 文件结构

```
.
├── script_director.py        # χ
├── cinematographer.py        # Γ (后端无关)
├── storyboard_maker.py       # ζ
├── diffusion_backend.py      # Γ real 后端 (SD + MV-Adapter)
├── diffusion_backend_stub.py # Γ stub 后端 (PIL 占位)
├── Makefile                  # 工作流封装
├── .env.example
├── requirements.txt          # 本地依赖
├── requirements-gpu.txt      # 云端 GPU 依赖
└── data/
    └── sample.txt            # 示例剧本
```

## 已知限制

- **只支持双人对话场景**。三人及以上场景会被 ζ 跳过并打 warning
- **场景切换 = 新场景**,不支持单 scene 内的空间变化
- **MV-Adapter 朝向语义反直觉**: `front_right`(az=45) 实际脸朝左,`front_left`(az=315) 实际脸朝右。SHOTS 表里已经做了反向映射
- **shot 3/4 暂未派发**: 模板生成但未被 timeline 使用,留给未来 LLM 选 shot 升级
- **Hunyuan3D-1 没接**: 双人简版用不到斜背视角,代码里保留接口签名
