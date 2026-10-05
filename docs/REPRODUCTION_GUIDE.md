# Huracan 复现指南

基于论文 **"Huracan: A Skillful End-To-End Data-Driven System for Ensemble Data
Assimilation and Weather Prediction"** (Ni et al., arXiv:2508.18486v2, 2025-08 提交 /
2026-06 修订) 整理。**官方未发布代码与权重**（截至 2026-10），本指南所有规格均标注
论文章节出处；论文未明确、复现时必须自行决定的项集中列于第 6 节，仓库现有选择列于
第 7 节。

> 标注约定：`(Sec 2.2)`、`(Table 3)`、`(A.2)` 等均指该论文的章节/表格；
> `[repo]` 指本仓库的实现选择（论文未规定）。

---

## 0. TL;DR —— 复现要素一页速查

| 要素 | 规格 | 出处 |
|---|---|---|
| 系统 | 两个独立模型：同化模型 + 预报模型，架构相同（仅编码器输入通道数不同） | Sec 2.2 |
| 骨干 | SFNO (Bonev et al. 2023) + 三项修改 | Sec 2.2 |
| 修改 ① | block 内一个 MLP 换成 Swin Transformer 块（源自 Kunyu, Ni 2023） | Sec 2.2 |
| 修改 ② | 谱滤波压缩：κ(l) = Σᵢ hᵢ(l)·κ̃ᵢ，L=179，k=15，参数降 12× | A.2 |
| 修改 ③ | 高斯噪声 → MLP → 条件 LayerNorm（AdaSpeech 风格） | Sec 2.2 |
| IO 约定 | 双时步输入 (T₀−6h, T₀) → 双时步输出 (T₀+6h, T₀+12h)，时间维展平 | Sec 2.2 |
| 状态变量 | 72 = z,t,u,v,q × 13 层 + 7 地面变量 | Table 2 |
| 观测通道 | 每时步 269 通道（论文正文口径，明细见 2.4 节的会计疑点） | Sec 2.2, Table 2 |
| 训练 | 6 阶段（det → stoch → fcst_ar / assim → joint → hybrid） | Sec 2.3, Table 3 |
| 关键超参 | batch 16，随机模型训练集成 M=2 | A.3 |
| EDA 推理 | 48 独立成员，2023-10-01 00Z ERA5 初始化，≥3 个月 spin-up | Sec 3.1 |
| 评估 | 2024 年 100 个起报时次 × 10 天；CRPS skill score / ESSR / 记分卡 | Sec 3.1 |

---

## 1. 系统总览

Huracan 是一个观测驱动的端到端集合预报系统，由两个模型耦合而成：

```
                ┌────────────────────────────────────────────────┐
  观测 (过去6h) │  同化模型 (EDA)                                  │
  ────────────►│  输入: (x_{T-6h}, x_T) + forcing + 观测 269ch×2 │
  背景 (上一循环)│  输出: 分析 x_{T+6h}, x_{T+12h}（残差增量）      │
                └───────────────┬────────────────────────────────┘
                          6h 循环 ×48 成员（仅噪声不同）
                                │ spin-up ≥ 3 个月后的集合平均 = "analysis"
                                ▼
                ┌────────────────────────────────────────────────┐
                │  预报模型                                        │
                │  输入: (x_{T-6h}, x_T) + forcing（无观测）       │
                │  输出: 10 天集合预报（每次调用 +12h）             │
                └────────────────────────────────────────────────┘
```

- 同化与预报**共享同一架构**，仅编码器输入通道数不同（Sec 2.2）。论文原话：
  同化模型本质上是"加了同化的预报模型"（a forecasting model with assimilation
  added on top）。
- 同化只需**一次模型调用**（orders of magnitude 快于 NWP 变分同化）（Sec 4）。
- 集合成员间无控制成员，48 个成员完全独立运行，扰动仅来自噪声嵌入（Sec 3.1）。
- Huracan 的 "analysis" 定义为 48 个 EDA 成员的**集合平均**（与 NWP 惯例一致）
  （Sec 3.1）。

---

## 2. 数据规格（Sec 2.1 + Table 1/2）

### 2.1 ERA5 与时间划分

| 用途 | 时间范围 | 出处 |
|---|---|---|
| 确定性/随机基模型、fcst_ar 训练 | **1979–2014 ERA5（36 年）** | Sec 2.3, Table 3 |
| assim / joint 训练 | **2010–2022 ERA5 + 观测** | Table 3 |
| 观测存档总跨度 | 2010–2024（= Sec 2.3 所称 "15-year data"，含验证/测试年） | Table 1, Sec 2.1 |
| 验证 / 测试 | 2023 / 2024 | Sec 2.1 |
| hybrid 微调 | 2018–2022 ECMWF HRES | Table 3 |

**模型变量（72 个，Table 2）**：

- 高空 65 = `z, t, u, v, q` × 13 层（论文未给出具体层列表 → 见第 6 节空白 G2）
- 地面 7 = `2t, 2d, 10u, 10v, tcc, tcwv, msl`
- 输入附加（每时步）：insolation 1 通道 + 常量 `z`(地形), `lsm` 2 通道

### 2.2 观测源全清单（Table 1）

| 类别 | 仪器 | 时段 | 数据集/格式 |
|---|---|---|---|
| 微波探测器 | NOAA 15–19 / Metop A–C / Aqua AMSU-A | 2010–2024 | L1b (NOAA CLASS / EUMETSAT / GES DISC) |
| | NOAA 15–17 AMSU-B | 2010–2014 | L1b (NOAA CLASS) |
| | NOAA 18,19 / Metop A–C MHS | 2010–2024 | L1b (NOAA CLASS / EUMETSAT) |
| | S-NPP, NOAA 20,21 ATMS | 2011–2024 | L1b (GES DISC / CLASS) |
| 高光谱红外 | Metop A–C IASI | 2013–2024 | L1c (EUMETSAT) |
| | S-NPP, NOAA 20,21 CrIS | 2012–2024 | L1b (GES DISC) |
| | Aqua AIRS | 2010–2024 | L1b (GES DISC) |
| 掩星 | GNSS-RO | 2010–2024 | L2 弯折角 (UCAR / AWS 开放数据) |
| 静止卫星成像仪 | GOES 16–19 ABI | 2017–2024 | L1b (AWS 开放) |
| | Himawari 8,9 AHI | 2015–2024 | L1b (AWS 开放) |
| | Meteosat 8–11 SEVIRI | 2010–2024 | L1.5 (EUMETSAT) |
| | Meteosat 8,9 IODC SEVIRI | 2017–2024 | L1.5 (EUMETSAT) |
| | GOES 11–15 IMAGER | 2010–2017 | GridSat-GOES (开放) |
| | 全球卫星合成 | 2010–2017 | GridSat-B1 (开放) |
| 地面观测 | 探空 Radiosondes | 2010–2024 | IGRA (开放) |
| | 陆地站点 Land stations | 2010–2024 | ISD (开放) |

论文**刻意省略**了 scatterometer、flight winds 等观测源（Sec 2.1）；选择依据是
NWP 的观测贡献诊断（Dahoui et al. 2017; Healy et al. 2024）。

获取渠道与注册要求速查见**附录 B**。

### 2.3 观测预处理管线（Sec 2.1，5 步，必须按序复现）

1. **质量控制**：剔除坏质量标记与物理不合理值（亮温 < 100 K 或 > 400 K）。
2. **自编码器压缩**：高光谱红外与 RO 数据统一格式后分别压到 **32** 与 **64**
   通道（实现细节见 2.5）。
3. **空间插值**：L1 卫星观测先从原生坐标双线性插值到**高分辨率中间网格**，再
   **守恒插值**到最终 1° 网格；已是网格的 GridSat 同样守恒插值；稀疏的 in-situ
   与 RO 嵌入**直接赋给最近网格点**，同格点多观测取平均。
4. **通道合并与时间分帧**：名义频率相近的卫星通道合并为一；所有观测归入等距
   时间帧，**每帧只保留离帧边界最近的观测**；卫星视角天顶角、观测与帧边界的
   时差等元数据作为附加通道加入。
5. **归一化与零填充**：归一化后，无观测的网格点置零。

注意（Sec 2.2）：观测通道挂在**每个输入时步**上，内容是"该时步之前 6 小时内发生
的全部观测"。因此 SEVIRI（小时级）每 6h 窗口有 6 个子帧，ATMS/IR（3 小时级）有
2 个子帧——这正是 Table 2 中 `6×11`、`2×23`、`2×32` 记法的含义。

### 2.4 每时步观测通道预算（Table 2 逐项）

| 通道数 | 内容 | 频次/备注 |
|---|---|---|
| 6×11 = 66 | SEVIRI 通道 1–11（含其他静止卫星的匹配通道，note 1） | 小时级 |
| 2×23 = 46 | ATMS 通道 1–22 + MHS 通道 1（含匹配通道，note 1） | 3 小时级 |
| 2×32 = 64 | 32 个高光谱 IR 嵌入 | 3 小时级 |
| 64 | 64 个 RO 嵌入 | — |
| 65 | 高空变量 in-situ 观测（= 5 变量 × 13 层） | — |
| 6 | 地面变量 in-situ 观测（除 tcwv） | — |
| 6×1 = 6 | 静止卫星天顶角 | — |
| 2×4 = 8 | MW 与 IR 探测器天顶角（note 2） | — |
| 2×4 = 8 | MW 与 IR 探测器时间差（note 2） | — |
| 1 | RO 剖面时间差 | — |

note 1：含其他仪器/数据集的匹配通道；AMSU-A、AMSU-B/MHS、ATMS、高光谱 IR 每族
仪器每时间帧合成一个通道。note 2：每仪器每帧一通道。

> **⚠️ 通道会计疑点**：上表逐项相加为 **334**，而论文正文 (Sec 2.2) 称
> "269 channels per time step"。334 − 65（高空 in-situ 通道）恰为 269，推测正文
> 口径可能不含 in-situ 高空列，或论文存在计数不一致。复现时建议按 Table 2 结构
> 组织通道、以程序统计实际通道数，不要硬凑 269。

### 2.5 高光谱 IR / RO 自编码器（A.1）

**输入制备**：

- 高光谱 IR：CrIS 光谱先 **apodize 到与 IASI 相同的处理级**；全部转亮温；统一
  重采样到 **0.125 cm⁻¹** 分辨率，共 **16,921** 通道。
- RO：弯折角剖面插值到 **999 个气压-高度层**（0.001–0.999 atm）；每层 6 个特征
  （弯折角、与最近网格点的坐标差、方位角等），拼接为 **5,994** 通道。

**自编码器**：

- 两个独立的 MLP 自编码器：**4 层编码器 + 4 层解码器**，bottleneck = 嵌入维
  （IR 32 / RO 64），**L1 重建损失**。
- 缺失通道（如非 IASI 仪器缺波段、未到达地面的掩星）**归一化后置 −5**。
- 训练完成后 Huracan 运行时只用编码器。

---

## 3. 模型架构（Sec 2.2 + A.2）

### 3.1 SFNO 骨干与三项修改

骨干是 Spherical Fourier Neural Operator（Bonev et al. 2023）。Huracan 的三处
实质修改：

**① Swin Transformer 替换 MLP（Fig. 2）**
每个 SFNO block 中的一个 MLP 被 Swin Transformer 块（Liu et al. 2021）替换。
该做法源自 Kunyu（Ni 2023），论文称显著提升网络建模能力与实际性能。论文只给出
block 结构图（Fig. 2），未给窗口大小、头数等超参 → 第 6 节空白 G1。

**② 谱滤波压缩（A.2）**
原始 SFNO 为每个度数 0 ≤ l ≤ L 配一个独立线性滤波器 κ̃_ϑ(l)。Huracan 的做法：

```
度数 l 的正弦编码(l) ⊕ 噪声嵌入 → MLP → h(l) ∈ R^k
κ(l) = Σ_{i=1..k} hᵢ(l) · κ̃ᵢ          （k 个共享可学习滤波器的加权和）
```

由此只有 k 个可学习滤波器而非 L+1 个。论文规模：1° 分辨率 → **L=179**，取
**k=15**，参数量降 **12 倍**，同时缓解过拟合。论文公式对 hᵢ(l) 无 softmax 约束
（无归一化要求）。

**③ 噪声嵌入 + 条件 LayerNorm（Sec 2.2）**
类似 AIFS-CRPS：随机高斯噪声经 MLP 处理生成噪声嵌入，作为每个网格点的
context embedding，送入 **条件 LayerNorm**（AdaSpeech, Chen et al. 2021）替换
原有标准 LayerNorm。这是集合离散度产生的机制。

### 3.2 双时步输入/输出（Sec 2.2）

- 输入 `(T₀−6h, T₀)`：每时步含 72 模型变量 + 3 forcing/常量；同化模型每时步
  另加观测通道（该时步前 6h 内的观测，见 2.3/2.4）。
- 输出 `(T₀+6h, T₀+12h)`，内部把时间维展平进通道（time flattened into
  channels）。
- 目的：稳定自回归推理、减少 rollout 微调所需的自回归步数。

### 3.3 同化模型的初始化（Sec 2.3）

同化模型 = 预报模型架构 + 编码器新增观测输入列。初始化时**继承随机基模型全部
权重，唯独编码器中对应观测通道的新列随机初始化**。

---

## 4. 训练配方（Sec 2.3 + Table 3 + A.3）

**全局设置（A.3）**：所有实验 **batch size = 16**；所有随机（stochastic）模型
训练时**集成成员 M = 2**。

阶段依赖与配方：

```
deterministic ──► stochastic ──┬──► fcst_ar (预报分支) ──┬──► joint (双模型)
 (L1, 无噪声)     (+噪声, CRPS)  │                        │      │
                                └──► assim (观测分支) ────┘      └─(可选)─► hybrid
```

| # | 阶段 | 训练数据 | Max LR | Schedule | 步数 | 关键设置（Sec 2.3） |
|---|---|---|---|---|---|---|
| 1 | deterministic 基模型 | 1979–2014 ERA5 | 5e-4 | cosine | 347k | L1 损失，无噪声 |
| 2 | stochastic 基模型 | 1979–2014 ERA5 | 2e-4 | cosine | 149k | 加噪声嵌入，CRPS 损失 |
| 3 | initial 预报模型 (fcst_ar) | 1979–2014 ERA5 | 3e-6 | constant | 14k | **6 步自回归微调**（rollout 窗口 3 天） |
| 4 | initial 同化模型 (assim) | 2010–2022 ERA5+Obs | 2e-4 | cosine | 46k | 从 stochastic 继承（观测列随机），**8 步自回归**（窗口 4 天）；每步初值与真值来自 ERA5、观测通道来自存档 |
| 5 | final Huracan (joint) | 2010–2022 ERA5+Obs | 3e-6 | cosine | 13k | **8 步同化 + 8 步预报联合微调**，两模型同时训练 |
| 6 | Huracan-Hybrid（可选基线） | 2018–2022 ECMWF HRES | 3e-6 | constant | 2.7k | 微调 initial 预报模型；推理时用 ECMWF ENS 集合初值 |

**设计依据（复现时容易忽略的动机）**：

- assim 的 8 步（4 天）rollout 窗口不是随意的——与旧背景误差衰减到可忽略所需
  时间一致（Berre 2019），迫使模型充分利用观测而非依赖 ERA5 初始背景
  （Sec 2.3）。
- joint 阶段 8+8 = 同化 4 天 + 预报 4 天，端到端闭环。
- hybrid 只是**对照基线**，不属于 Huracan 系统本体；不复现它不影响主结果。

---

## 5. 推理与评估协议（Sec 3.1）

### 5.1 EDA 循环

- **48 个完全独立的 EDA 成员**，每个成员从**同一 ERA5 初始场**初始化：
  **2023 年 10 月 1 日 00 UTC**。
- 成员间唯一差异是噪声嵌入的实现（各成员独立采样）。
- **spin-up ≥ 3 个月**（论文认为 3 个月后初始条件来自 NWP 的影响可忽略），之后
  进入 2024 测试年。`[repo]` paper.yaml 取 720 个循环（3 月 × 4 循环/日）。
- 每个同化循环 = 1 次模型调用（输入上一循环输出 + 新观测）。

### 5.2 评估设置

| 项 | 规格 |
|---|---|
| 起报时次 | 2024 年内 **100 个**（1 月 1 日 – 12 月 17 日，受 ECMWF hindcast 可用性限制） |
| 预报时长 | 10 天（20 个 6h 输出） |
| 真值 | **各系统对自己的 analysis** 评估（Huracan 的 analysis = 48 成员集合平均；ECMWF 用自家 analysis） |
| 分辨率匹配 | ECMWF ENS 的初值/预报/分析**守恒降采样**到 1° |
| 指标 | CRPS、CRPS skill score（GraphCast 记号）、ESSR（ensemble spread-skill ratio）、记分卡 |
| 评分变量 | z, t, q @ 200/500/700/850 hPa + 2t, 2d, 10u, 10v, msl |
| 主要结论 | vs ECMWF ENS：697 个变量×时效组合中 80.2% 达到（差距 <2% 即算持平）或更优；vs Huracan-Hybrid：65.1% |

### 5.3 论文自述的行为特征（复现对照点）

- **ESSR 前几个时效偏高**：方法学差异（ENS 由控制成员扰动生成，Huracan 无控制
  成员、48 成员独立，集合均值更接近真期望值）——复现时出现属正常。
- **ESSR 在 3–5 天偏低**：论文归因于过拟合（训练年 2017 的 ESSR 全程 > 1）。
- **训练/测试年差距大**：论文承认训练年指标远好于测试年（Sec 4），复现时应同时
  报告训练年与测试年指标。
- 公平性注意：ECMWF 的 00 UTC 初值实际包含了直到 03 UTC 的观测（即"未来"观测），
  Huracan 不使用未来观测（Sec 4）。

---

## 6. 论文未明确处清单（复现必读）

以下各项论文**没有规定**，复现必须自行决策。表中同时给出本仓库的选择与理由，
**请勿把仓库选择当成论文要求**。

| # | 空白点 | 论文状态 | 仓库选择 `[repo]` | 理由 |
|---|---|---|---|---|
| G1 | hidden 维度 / block 数 / 注意力头数 / Swin 窗口 / mlp_ratio | 未给（仅给 Fig. 2 结构） | paper.yaml: 512 / 12 / 8 / 7 / 4.0；pilot: 192 / 6 / 3 / 8 / 4.0 | 常规 SFNO/Swin 量级估计 |
| G2 | 13 个气压层具体列表 | 未列 | 50,100,150,200,250,300,400,500,600,700,850,925,1000 hPa | ERA5 标准 13 层子集；覆盖评估层 200/500/700/850 |
| G3 | 优化器 / weight decay / warmup / 梯度裁剪 | 未给 | AdamW, wd=0.01, 线性 warmup 2%, grad clip 1.0 | 天气 ML 模型通行做法 |
| G4 | 噪声嵌入维度与注入方式 | 仅说"高斯噪声 + MLP" | 每像素 N(0,1)，noise_dim 16（pilot 8），嵌入维 128（pilot 64） | 参考 AIFS-CRPS 量级 |
| G5 | CRPS 估计器形式（fair 或 biased） | 未给 | fair CRPS：E\|X−y\| − 0.5·E_{m≠m'}\|X−X'\| | GenCast/AIFS-CRPS 惯例（论文引用同源） |
| G6 | 归一化统计 | 仅说"归一化" | 逐通道 mean/std（训练期数据统计），观测复用对应状态变量统计 | 最简且与零填充约定兼容 |
| G7 | HEALPix 排列（NESTED/RING） | 未给 | NESTED | 面式 Swin 要求像素能 reshape 成 12×n×n 图像 |
| G8 | insolation 计算式 | 未给 | 解析式：太阳赤纬 23.45°·sin(2π(284+doy)/365)，S=1361 W/m²，cos(zenith) 截断 | 标准天文公式 |
| G9 | spin-up 具体长度 | 仅"≥3 个月" | paper: 720 循环（=3 个月）；pilot: 120 | 下限取整 |
| G10 | rollout 损失如何聚合（求和/平均/加权） | 未给 | 对 rollout 内所有输出步平均 | 无信息时的默认 |
| G11 | Swin 面边界的 shift wrap 方式 | 未给 | 面内 wrap（非真实 HEALPix 邻接） | 实现简化，已标注 |
| G12 | SHT 后端 | 未给（SFNO 原实现为 torch-harmonics） | 自实现实基最小二乘 SHT（精确、O(npix²)）；nside=64 需换 torch-harmonics（接口已对齐） | pilot 规模下自包含可跑 |
| G13 | 观测通道总数（269 vs 334） | 正文与 Table 2 不一致 | 按 Table 2 结构实现，pilot 仅 in-situ 部分（70 通道） | 见 2.4 节疑点 |
| G14 | 模型参数量 | 未给 | 未声明（paper.yaml 规模下可自行统计） | — |
| G15 | EDA 成员扰动来源 | 仅"独立成员" | 每成员独立噪声种子（1000+m） | 与噪声嵌入机制一致 |

---

## 7. 论文规格 ↔ 本仓库实现对照

### 7.1 配置对照表

| 项 | 论文 | `configs/paper.yaml` | `configs/pilot.yaml` |
|---|---|---|---|
| nside | 64（~1°，由 1° 分辨率隐含） | 64 | 16（~3.7°） |
| lmax | 179（A.2） | 179 | 32 |
| k_filters | 15（A.2） | 15 | 8 |
| hidden / blocks / heads / window | 未给（G1） | 512 / 12 / 8 / 7 | 192 / 6 / 3 / 8 |
| noise_dim / noise_embed_dim | 未给（G4） | 16 / 128 | 8 / 64 |
| batch / 训练成员 | 16 / 2（A.3） | 16 / 2 | 8 / 2 |
| 训练步数（6 阶段） | 347k/149k/14k/46k/13k/2.7k | 同论文 | 40k/15k/3k/5k/2k/1k |
| LR（6 阶段） | 5e-4/2e-4/3e-6/2e-4/3e-6/3e-6 | 同论文 | 同论文 |
| rollout（fcst/assim/joint） | 6 / 8 / 8+8 | 同论文 | 同论文 |
| 观测通道 | 269（Table 2） | 目标 269（卫星部分需注册数据） | 70（IGRA+ISD in-situ） |
| EDA 成员 / spin-up / 时效 | 48 / ≥3 月 / 10 天 | 48 / 720 / 20 步 | 8 / 120 / 20 步 |
| ERA5 年份 | 36 年 + 13 年 | 1979–2014 + 2010–2022 | 2010–2013 |

### 7.2 模块映射

| 论文概念 | 出处 | 仓库实现 |
|---|---|---|
| HEALPix 网格 / forcing | Sec 2.2 | `huracan/grid.py`（NESTED 像素中心、insolation、旋转工具） |
| 球谐变换 | SFNO 需要 | `huracan/sht.py`（`MatrixSHT`，实基最小二乘，往返误差 ~1e-6） |
| 噪声嵌入 → 条件 LN | Sec 2.2 | `huracan/modules.py`：`NoiseEmbedder`、`ConditionalLayerNorm` |
| 谱滤波压缩 κ(l)=Σhᵢ(l)κ̃ᵢ | A.2 | `huracan/modules.py`：`CompressedSpectralFilter` |
| Swin 替换 MLP | Sec 2.2, Fig. 2 | `huracan/modules.py`：`SwinFaceBlock` + `WindowAttention`（W-MSA + SW-MSA + MLP，相对位置偏置） |
| 完整 SFNO block | Fig. 2 | `huracan/modules.py`：`HuracanSFNOBlock`（x + ISHT(Filter(SHT(CLN(x))))，再 x + Swin(CLN(x))） |
| 双时步 IO / 残差增量 | Sec 2.2 | `huracan/model.py`：`HuracanSFNO`（`step()` 输出对 x_T 的残差；输出层零初始化） |
| 同化 = 预报 + 观测列 | Sec 2.3 | `huracan/model.py`：`build_models`（use_obs）；`train/trainer.py`：`load_with_new_columns`（编码器仅新列随机） |
| 观测自编码器 | A.1 | `huracan/autoencoder.py`：`ObsAutoencoder`（4+4 层，L1，缺失=−5；已在合成数据上冒烟测试，等待真实卫星数据） |
| 分阶段训练 / Table 3 | Sec 2.3 | `huracan/train/trainer.py`：`HuracanTrainer`（阶段链式 init、cosine+warmup、constant 选项） |
| L1 / CRPS 损失 | Sec 2.3 | `huracan/train/losses.py`（fair CRPS，M=2 时退化为 mean\|·\|−0.5\|x₁−x₂\|） |
| 数据集/归一化 | Sec 2.1 | `huracan/data/`：`dataset.py`（连续 2k+2 帧块）、`norm.py`（观测复用状态统计、零填充）、`obs.py`（IGRA/ISD 解析）、`interp.py` |
| EDA 循环 + 端到端预报 | Sec 3.1 | `huracan/inference.py`：`run_eda`（spin-up → 预报，逐成员独立种子） |
| CRPS / ESSR / 记分卡 | Sec 3.1 | `huracan/eval/`：`metrics.py`、`scorecard.py` |
| 入口脚本 | — | `scripts/`：`prepare_era5_wb2.py`、`download_obs.py`、`prepare_obs.py`、`train.py`、`run_eda.py` |

### 7.3 已知偏差与简化（诚实清单）

1. **无卫星观测**：IASI/CrIS/ATMS/SEVIRI 等 L1B 需逐个注册申请（附录 B），
   pilot 观测只有探空 + 地面站（Table 2 的 in-situ 部分，70/269 通道）。
   `autoencoder.py` 接口就绪，拿到数据即可接入。
2. **无 hybrid 对照**：需 ECMWF HRES/ENS 数据（MARS 许可），pilot 跳过
   （hybrid 阶段代码保留）。
3. **SHT 后端**：自实现最小二乘 SHT 精确但 O(npix²)，nside=64 时必须换
   torch-harmonics。
4. **Swin shift 面内 wrap**：非真实 HEALPix 面间邻接（G11）。
5. **度数权重无 softmax**：与论文公式一致（A.2 无归一化项），非简化。
6. **评估真值**：pilot 用 ERA5 对所有系统评估；论文对各自 analysis 评估
   （Huracan = 48 成员集合平均）。
7. **规模整体缩减**：见 7.1 表 pilot 列；训练步数约为论文的 1/10 以下。

---

## 8. 复现路线图

### 8.1 Level 0 —— 冒烟验证（CPU，~1 分钟）

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/test_smoke.py   # 合成数据跑通全链路：训练→EDA→评估
```

目的：验证架构、训练循环、EDA 循环、指标代码正确性，与数据无关。

### 8.2 Level 1 —— 单卡 pilot（RTX 4090，4–6 h）

```bash
# 数据（免注册）：WeatherBench2 ERA5 1°/6h + IGRA/ISD
.venv/bin/python scripts/prepare_era5_wb2.py --config configs/pilot.yaml \
    --years 2010 2011 2012 2013 --workers 6
.venv/bin/python scripts/download_obs.py --years 2010 2011 2012 2013 \
    --n-igra 400 --n-isd 1500
.venv/bin/python scripts/prepare_obs.py --config configs/pilot.yaml
# 训练 + EDA 评估
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/train.py \
    --config configs/pilot.yaml --phase all
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_eda.py --config configs/pilot.yaml
```

能验证：真实数据管线、全部 6 个训练阶段、EDA 循环、CRPS/ESSR 曲线形态。
不能验证：论文精度（规模差 1–2 个数量级）、卫星同化、hybrid 对照。

### 8.3 Level 2 —— 论文规模 scale-up 清单

- [ ] **多 GPU 训练**（paper.yaml 规模，36 年数据、347k 步；需分布式改造）
- [ ] **SHT 换 torch-harmonics**（nside=64；`MatrixSHT.forward/inverse` 接口已对齐）
- [ ] **卫星数据申请**（按附录 B 矩阵逐个注册：EUMETSAT / NASA GES DISC /
      NOAA CLASS；GridSat、GOES/Himawari AWS、GNSS-RO UCAR 开放）
- [ ] **自编码器实跑**：A.1 输入制备（CrIS apodize → 0.125 cm⁻¹ 重采样；RO →
      999 层 × 6 特征）+ 两个 AE 预训练，再接主训练
- [ ] **观测全管线**：2.3 节 5 步（含中间网格双线性 + 守恒插值、时间分帧、
      元数据通道）——pilot 只实现了第 3 步的"最近格点"分支与第 5 步
- [ ] **ECMWF 数据**（hybrid 基线用，需 MARS/许可）
- [ ] **EDA 推理算力**：48 成员 × 720 spin-up 循环 × 100 起报时次
- [ ] 对照数据：ECMWF ENS 预报/analysis 守恒降采样到 1°

### 8.4 风险与未知项

| 风险 | 说明 | 缓解 |
|---|---|---|
| 通道会计 | 269 vs 334（2.4 节） | 按 Table 2 结构实现，程序统计实际数 |
| 层列表 G2 | 论文未列 13 层 | 仓库列表覆盖评估层，风险低 |
| SHT 后端切换 | 最小二乘 vs torch-harmonics 数值差异 | 切换前做往返误差与训练收敛对照 |
| 过拟合 | 论文自述训练/测试年差距大（Sec 4） | 同时报告训练年与测试年指标 |
| 观测分帧细节 | "每帧取离边界最近"的具体实现影响分布 | 对照 Fig. 1 样例图目检 |
| CRPS 形式 | fair/biased 对 M=2 训练有影响（G5） | 仓库用 fair；如结果异常可对照 |
| 卫星数据周期 | 各仪器起始年不同（Table 1），2010–12 缺 IASI/ATMS | 按论文同口径：缺失置零/−5 约定 |

---

## 附录 A：变量与层级速查

**高空变量（×13 层）**：`z` geopotential, `t` temperature, `u`/`v` wind
components, `q` specific humidity
**层列表 `[repo]`**：50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925,
1000 hPa（论文 G2 空白）
**地面变量**：`2t` 2m temperature, `2d` 2m dewpoint, `10u`/`10v` 10m wind,
`tcc` total cloud cover, `tcwv` total column water vapor, `msl` mean sea level
pressure
**评估用子集（Sec 3.1）**：z/t/q @ 200/500/700/850 + 2t, 2d, 10u, 10v, msl
（tcwv、tcc 不进记分卡）

## 附录 B：数据源获取速查

| 数据 | 提供方 | 获取方式 | 注册门槛 |
|---|---|---|---|
| AMSU-A/B, MHS (NOAA) | NOAA CLASS | aev.class.noaa.gov（TOVS 族） | 免费注册 |
| ATMS SDR (NOAA 20/21) | NOAA CLASS | 同上（ATMS_SDR 族） | 免费注册 |
| ATMS/CrIS/AIRS (SNPP/JPSS/Aqua) | NASA GES DISC | disc.gsfc.nasa.gov（SNPPATMSL1B、SNDRJ1CrISL1B、AIRIBRAD 等数据集） | 免费注册 |
| AMSU-A/MHS/IASI L1c, SEVIRI L1.5 | EUMETSAT | navigator.eumetsat.int（EO:EUM:DAT 系列） | 注册 + 许可协议 |
| GNSS-RO L2 | UCAR | registry.opendata.aws/gnss-ro-opendata | 开放 |
| GOES 16–19 ABI | NOAA | registry.opendata.aws/noaa-goes/ | 开放（AWS） |
| Himawari 8/9 AHI | NOAA/JMA | registry.opendata.aws/noaa-himawari/ | 开放（AWS） |
| GridSat-GOES / B1 | NOAA NCEI | NCEI 官网 | 开放 |
| IGRA 探空 | NOAA NCEI | ncei.noaa.gov（IGRA v2） | 开放 |
| ISD 地面站 | NOAA NCEI | ncei.noaa.gov（ISD） | 开放 |
| ERA5 | Copernicus CDS / WeatherBench2 | cds.climate.copernicus.eu 或 GCS 公开 zarr | CDS 需注册；WB2 匿名 |
| ECMWF ENS/HRES | ECMWF MARS | ecmwf.int | 需许可（学术可申请） |

---

*本指南对应论文 arXiv:2508.18486v2；仓库状态：pilot 已实跑（见 README），paper
规模配置就绪待数据与算力。如论文日后官方开源，以官方实现为准复核第 6 节各项空白。*
