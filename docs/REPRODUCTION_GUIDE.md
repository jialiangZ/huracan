# Huracan 复现指南（修订版 / Faithful-Reproduction Edition）

> 对应论文：**Huracan: A Skillful End-To-End Data-Driven System for Ensemble Data Assimilation and Weather Prediction**  
> Ni et al., arXiv:2508.18486v2  
> 本文档基于论文 v2（2026-06 修订）重新核对，并对旧版复现指南中的高风险假设进行了修正。

---

## 文档状态与使用原则

截至 2026-10，论文没有公开官方训练代码、模型权重或完整配置。因此，**严格逐位复现（bitwise reproduction）目前不可实现**。本指南的目标是：

1. 完整收集论文中**明确给出的事实**；
2. 将**高置信推断**与论文事实严格分开；
3. 将所有**仓库工程选择**显式标记，不把它们伪装成论文要求；
4. 把会显著影响结果、但论文未公开的细节集中列入“不可恢复项 / Open Questions”；
5. 给出一条可以逐步验证的复现路线。

本文统一使用以下标记：

- **[PAPER]**：论文正文、表格或附录明确给出；
- **[INFERRED]**：由论文内部约束或其引用实现高置信推断，但论文未逐字说明；
- **[REPO]**：为了可运行复现而采用的实现选择，不能声称是 Huracan 原始设置；
- **[OPEN]**：论文未公开，无法从论文恢复；
- **[RISK]**：若选择错误，可能显著影响复现结果。

> **最重要的修订**
>
> 旧版指南曾将 1° SFNO 具体化为 **HEALPix nside=64 / NESTED**。论文并未使用或提及 HEALPix，这一假设应删除。  
> 新版默认采用 **规则纬经度 / equiangular SFNO 路线**，但仍将最终网格节点定义保留为 [OPEN]。  
> 同时修正 autoregressive step：论文的 6-step = 3 days、8-step = 4 days，意味着一次 autoregressive model call 推进 **12 h**，同时输出两个 6 h 场。

---

# 0. TL;DR —— 一页复现规格

| 项目 | 规格 | 状态 |
|---|---|---|
| 系统 | 两个独立模型：EDA 同化模型 + 预报模型；架构相同，仅 encoder 输入通道数不同 | [PAPER] |
| 主干 | SFNO（Bonev et al.） | [PAPER] |
| 修改 1 | SFNO block 中一个 MLP 替换为 Swin Transformer block | [PAPER] |
| 修改 2 | 谱滤波压缩：\(\kappa(l)=\sum_i h_i(l)\tilde{\kappa}_i\)，\(L=179,k=15\) | [PAPER] |
| 修改 3 | Gaussian noise → MLP → context embedding → conditional LayerNorm | [PAPER] |
| 空间分辨率 | 1° | [PAPER] |
| 空间网格 | 规则球面纬经度 / equiangular 路线最合理；精确节点仍未知 | [INFERRED]/[OPEN] |
| 输入 | \((T_0-6h,T_0)\) | [PAPER] |
| 输出 | \((T_0+6h,T_0+12h)\) | [PAPER] |
| AR 语义 | 1 autoregressive call 推进 12 h，同时产生两个 6 h 输出 | [INFERRED，论文 6-step=3d 强约束] |
| 状态变量 | 72 = 5 高空变量 × 13 层 + 7 地面变量 | [PAPER] |
| forcing/constants | insolation + orography \(z\) + land-sea mask | [PAPER] |
| 观测通道 | 正文称 269 / time step；Table 2 按条目相加为 334 | [PAPER，不一致] |
| 基础训练 | 1979–2014 ERA5；deterministic → stochastic | [PAPER] |
| DA 训练 | 2010–2022 ERA5 + Obs | [PAPER] |
| 训练 batch | 16 | [PAPER] |
| stochastic 训练 ensemble size | 2 | [PAPER] |
| fcst AR | 6 calls = 3 days | [PAPER] |
| assimilation AR | 8 calls = 4 days | [PAPER] |
| joint | 8 assimilation calls + 8 forecast calls | [PAPER] |
| EDA 成员 | 48 independent members | [PAPER] |
| EDA 起点 | 2023-10-01 00 UTC，同一 ERA5 初值 | [PAPER] |
| spin-up | 至少 3 个月 | [PAPER] |
| 测试 | 2024，100 个 issue times，10 lead days | [PAPER] |
| 主指标 | CRPS / CRPS skill score / ESSR | [PAPER] |
| 评分变量 | 17 个变量层组合 | [PAPER] |
| 697 个组合 | \(17\times41=697\)，与 0–240 h 每 6 h 共 41 个 lead 完全一致 | [INFERRED] |

---

# 1. 系统总览

Huracan 是一个**端到端集合数据同化 + 集合天气预报系统**。

```text
                             ┌───────────────────────────────────────┐
observations in past windows │        Assimilation model (EDA)       │
────────────────────────────►│  state + forcing + observations       │
member-specific background   │                                       │
────────────────────────────►│  (t-6h, t) -> (t+6h, t+12h)          │
                             └──────────────────┬────────────────────┘
                                                │
                                  autoregressive EDA cycling
                                  48 independent members
                                                │
                                                ▼
                                  Huracan analysis = ensemble mean
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │          Forecasting model             │
                             │  state + forcing; no observations      │
                             │  (t-6h, t) -> (t+6h, t+12h)          │
                             └───────────────────────────────────────┘
```

### 1.1 两个模型

**[PAPER]**

- assimilation model 和 forecasting model 是两个独立模型；
- 架构相同；
- 区别只在 encoder 输入通道数；
- assimilation model 在状态与 forcing 之外还接收观测通道；
- 论文称 assimilation model 可以理解成 “a forecasting model with assimilation added on top”。

### 1.2 一个容易混淆的概念：6 h 输出 ≠ 6 h autoregressive call

模型一次调用：

\[
(x_{t-6},x_t)\rightarrow(\hat x_{t+6},\hat x_{t+12}).
\]

若自回归继续：

\[
(\hat x_{t+6},\hat x_{t+12})
\rightarrow
(\hat x_{t+18},\hat x_{t+24}).
\]

因此：

- output temporal spacing = **6 h**；
- one autoregressive model call = **12 h advancement**；
- one call produces **2 forecast fields**。

这个解释由论文两条信息共同强约束：

- 6-step forecast rollout = 3 days；
- 8-step assimilation rollout = 4 days。

所以：

\[
6\times12h=72h=3d,\qquad
8\times12h=96h=4d.
\]

> 代码里建议明确区分：
>
> - `output_interval_hours = 6`
> - `ar_call_advance_hours = 12`
> - `outputs_per_call = 2`

---

# 2. 数据规格

## 2.1 数据时间划分

| 用途 | 数据 | 时间 |
|---|---|---|
| deterministic base | ERA5 | 1979–2014 |
| stochastic base | ERA5 | 1979–2014 |
| initial forecast AR | ERA5 | 1979–2014 |
| initial assimilation | ERA5 + observations | 2010–2022 |
| final Huracan joint | ERA5 + observations | 2010–2022 |
| validation | — | 2023 |
| test | — | 2024 |
| hybrid forecast fine-tune | ECMWF HRES | 2018–2022 |

**[PAPER]** 论文称观测档案跨度约 15 年，表中源数据覆盖至 2024，但主 DA 训练为 2010–2022，2023/2024 分别保留验证/测试。

---

## 2.2 状态变量

Huracan 同化并预测 72 个变量。

### 高空：65 channels

\[
z,t,u,v,q \times 13\text{ pressure levels}
\]

其中：

- `z`: geopotential
- `t`: temperature
- `u`: zonal wind
- `v`: meridional wind
- `q`: specific humidity

**[OPEN] 13 个 pressure levels 的完整列表未在论文列出。**

### 地面：7 channels

- `2t`
- `2d`
- `10u`
- `10v`
- `tcc`
- `tcwv`
- `msl`

### 输入专用

每个 input time step 额外包含：

- insolation × 1
- constant orography/geopotential `z` × 1
- land-sea mask `lsm` × 1

因此 forecast model 的每时间步最少有：

\[
72+3=75
\]

个输入通道（时间维随后 flatten）。

---

## 2.3 推荐 pressure-level 选择

论文没有列出 13 层，所以任何列表都属于 [REPO]。

可采用：

```text
50, 100, 150, 200, 250, 300, 400,
500, 600, 700, 850, 925, 1000 hPa
```

优点：

- 覆盖论文评分层 200 / 500 / 700 / 850 hPa；
- 是常见 ERA5 子集；
- 兼顾平流层至近地面。

但必须记录：

> **[RISK] 这不是论文公布的 13 层列表。若未来官方代码给出不同层级，需整体替换数据、模型通道与 in-situ vertical mapping。**

---

# 3. 观测源

## 3.1 微波探测器

| 仪器 | 时段 | 格式 |
|---|---|---|
| NOAA 15–19 AMSU-A | 2010–2024 | L1b |
| Metop A–C AMSU-A | 2010–2024 | L1b |
| Aqua AMSU-A | 2010–2024 | L1b |
| NOAA 15–17 AMSU-B | 2010–2014 | L1b |
| NOAA 18,19 MHS | 2010–2024 | L1b |
| Metop A–C MHS | 2010–2024 | L1b |
| S-NPP / NOAA 20,21 ATMS | 2011–2024 | L1b |

## 3.2 高光谱红外

| 仪器 | 时段 | 格式 |
|---|---|---|
| Metop A–C IASI | 2013–2024 | L1c |
| S-NPP / NOAA 20,21 CrIS | 2012–2024 | L1b |
| Aqua AIRS | 2010–2024 | L1b |

## 3.3 GNSS-RO

- 2010–2024
- Level 2 bending-angle profiles

## 3.4 静止卫星

| 仪器/数据 | 时段 |
|---|---|
| GOES 16–19 ABI | 2017–2024 |
| Himawari 8/9 AHI | 2015–2024 |
| Meteosat 8–11 SEVIRI | 2010–2024 |
| Meteosat 8/9 IODC SEVIRI | 2017–2024 |
| GOES 11–15 IMAGER / GridSat-GOES | 2010–2017 |
| GridSat-B1 global composites | 2010–2017 |

## 3.5 In-situ

- Radiosondes: IGRA
- Land stations: ISD

## 3.6 明确省略的观测

论文明确说明仍省略了一些可能有价值的来源，例如：

- scatterometers
- flight winds

所以 pilot 不应把这些额外观测混入后再声称是 paper-faithful Huracan。

---

# 4. 观测预处理

论文给出 5 个主要步骤。

## 4.1 Step 1 — Quality control

**[PAPER]**

- 去除 bad quality flags；
- 去除 physically implausible brightness temperatures：
  \[
  T_B <100K \quad\text{or}\quad T_B>400K.
  \]

**[OPEN]**

论文没有给出每个 instrument/product 的具体 flag bit 规则，例如：

- calibration flags
- scan-edge rejection
- cloud flags
- saturation
- moon contamination
- product-specific QC masks

因此当前只能复现**QC 原则**，不能复原作者精确 QC recipe。

---

## 4.2 Step 2 — IR / RO autoencoder compression

- hyperspectral IR → 32 channels
- GNSS-RO → 64 channels

详见第 6 节。

---

## 4.3 Step 3 — Spatial interpolation

### gridded / L1 satellite

**[PAPER]**

1. native satellite coordinates
2. bilinear interpolation → higher-resolution intermediate grid
3. conservative interpolation → final 1° grid

GridSat 数据已经 gridded，直接进入 conservative interpolation。

### sparse data

对：

- in-situ
- radio-occultation embeddings

直接分配到 nearest grid point；若同格点有多个观测，则 average。

### 未公开细节

- intermediate grid resolution
- intermediate grid coordinates
- conservative-remapping library
- mask handling
- longitude wrap
- polar handling
- area definition

这些全部属于 [OPEN]。

---

## 4.4 Step 4 — Channel merging + temporal framing

**[PAPER]**

- nominal frequency 相近的 satellite channels 合并；
- 所有观测分入 equally-spaced time frames；
- 只保留距 frame boundary 最近的观测；
- satellite viewing angle 等 metadata 加为额外通道；
- observation timestamp 与 frame boundary 的时间差也作为通道。

### 一个 input time step 对应多少历史观测？

同化模型在某个 input time \(t\) 接收：

> **该时间点之前 6 小时内的观测。**

因此论文 Table 2 中：

- `6 × 11`：1 h temporal frames × 6
- `2 × 23`：3 h temporal frames × 2
- `2 × 32`：3 h temporal frames × 2

### [OPEN] temporal frame 的精确定义

例如 3 h channels：

- frame boundary 是 \(-6,-3,0\)？
- interval 是 \([-6,-3)\) 与 \([-3,0)\)？
- 对等距 timestamp tie 怎么处理？
- 多 orbit / 多 instrument overlap 怎么 reduce？

论文均未说明。

---

## 4.5 Step 5 — Normalization + zero fill

**[PAPER]**

- observations 先 normalized；
- 没有 observation 的 grid points 设为 zero。

### 强制区分两个 zero

这意味着推荐数据结构需要显式保证：

```text
normalized observed value ~= arbitrary real number
missing observation = exactly 0
```

但论文没有给 normalization 方式，所以 0 是否与真实 normalized value 冲突、作者是否使用额外 availability mask，均未公开。

---

# 5. 观测通道预算与 269/334 矛盾

论文 Table 2：

| 通道数 | 内容 |
|---:|---|
| \(6\times11=66\) | SEVIRI 1–11 + matching channels |
| \(2\times23=46\) | ATMS 1–22 + MHS 1 |
| \(2\times32=64\) | hyperspectral IR embeddings |
| 64 | radio-occultation embeddings |
| 65 | upper-air in-situ |
| 6 | surface in-situ except tcwv |
| \(6\times1=6\) | geostationary zenith angle |
| \(2\times4=8\) | MW/IR zenith angles |
| \(2\times4=8\) | MW/IR time differences |
| 1 | RO profile time difference |

逐项相加：

\[
66+46+64+64+65+6+6+8+8+1=334.
\]

但论文正文明确称：

\[
269\text{ observation channels per time step}.
\]

而：

\[
334-65=269.
\]

因此最明显的可能性是：

> 正文的 269 没有把 65 个 upper-air in-situ channels 计入，或者 Table 2 / 正文存在计数不一致。

### 复现原则

**不要硬凑 269。**

推荐：

```yaml
observations:
  layout: table2
  expected_channels_from_text: 269
  expected_channels_from_table_sum: 334
  enforce_channel_count: false
```

程序启动时输出全部 channel accounting。

在官方实现公布前，这项必须保留为：

> **[OPEN][RISK-HIGH] exact assimilation encoder observation-channel count**

---

# 6. Hyperspectral IR / GNSS-RO Autoencoders

## 6.1 Hyperspectral IR input

**[PAPER]**

1. CrIS spectra 先 apodize，以匹配 IASI processing level；
2. 所有 hyperspectral spectra 转为 brightness temperature；
3. upsample 到：
   \[
   0.125\ \mathrm{cm}^{-1}
   \]
4. 总通道：
   \[
   16,921.
   \]

最终：

\[
16921 \rightarrow AE \rightarrow 32.
\]

---

## 6.2 GNSS-RO input

**[PAPER]**

- bending-angle profiles 插值到 999 pressure-altitude levels；
- 范围：
  \[
  0.001\text{ atm}\rightarrow0.999\text{ atm}
  \]
- 每 level 计算 6 个 features；
- 论文举例包括：
  - bending angle
  - coordinate difference from nearest grid point
  - angle of orientation
- concatenate：

\[
999\times6=5994.
\]

最终：

\[
5994\rightarrow AE\rightarrow64.
\]

> **[OPEN][RISK-HIGH] 六个 RO features 的完整列表并未给出。**

---

## 6.3 AE architecture

**[PAPER]**

IR 与 RO 使用两个独立的 MLP autoencoders：

- 4-layer encoder
- 4-layer decoder
- bottleneck = 32 / 64
- L1 reconstruction loss

missing channels：

\[
x_{\rm missing}=-5
\]

是在 normalization **之后**设定。

训练完成后 Huracan 主系统只使用 encoder。

---

## 6.4 AE 无法从论文恢复的细节

以下全部 [OPEN]：

- hidden widths
- activation
- normalization layer
- dropout
- optimizer
- learning rate
- batch size
- training steps/epochs
- train/val split
- sampling strategy
- sensor balancing
- spectral-channel normalization statistics
- RO feature normalization
- decoder output activation
- weight initialization

因此当前仓库的任何 AE hidden widths 都必须标 [REPO]。

---

# 7. 空间网格与 SHT —— 修订重点

## 7.1 论文真正说明了什么

**[PAPER]**

- Huracan output resolution = 1 degree；
- backbone = SFNO；
- filter-compression appendix 写：
  \[
  L=179
  \]
  且称 “1-degree resolution implies \(L=179\)”。

论文**没有**写：

- HEALPix
- nside
- NESTED/RING
- 12 faces

因此旧版中的这些设定必须删除为 paper specification。

---

## 7.2 推荐解释

SFNO 原始生态使用 spherical harmonic transform，`torch-harmonics` 的标准示例直接针对：

```text
[nlat, nlon]
regular spherical grid
```

并提供 `grid="equiangular"`。

结合：

\[
1^\circ,\quad L=179,
\]

最合理的高置信路线是：

> **[INFERRED] Huracan 使用与 SFNO/SHT 兼容的规则纬经度或等角球面网格，而不是 HEALPix。**

推荐 scale-up 起点：

```yaml
grid:
  family: latlon
  sht_grid: equiangular
  nlat: 180          # REPO / high-confidence inference
  nlon: 360          # REPO / high-confidence inference
  lmax: 179          # PAPER
```

### 但仍然不能声称 180×360 是论文事实

以下仍 [OPEN]：

- 180 vs 181 latitude samples
- poles 是否包含
- latitude centers / boundaries
- north-to-south ordering
- longitude origin
- final ERA5 remapping exactness
- SHT normalization convention

所以建议配置名：

```text
configs/scaleup_inferred.yaml
```

而不是 `paper.yaml`。

---

## 7.3 禁止保留的旧假设

以下不应再出现在 faithful reproduction config：

```text
HEALPix nside=64
NESTED
RING
SwinFaceBlock
12-face reshape
face-internal wrap
HEALPix pixel center
HEALPix MatrixSHT
```

除非未来得到官方证据。

---

# 8. 模型架构

## 8.1 两模型共架构

**[PAPER]**

Assimilation / forecast models：

- identical architecture
- encoder input channels different

---

## 8.2 SFNO modification 1 — Swin block

**[PAPER]**

在 SFNO blocks 中，一个 MLP 被 Swin Transformer block 替换。

**[OPEN]**

- hidden width
- number of blocks
- attention heads
- Swin window size
- shift size
- MLP ratio
- relative-position setup
- drop path
- dropout
- exact norm ordering

### 新版实现原则

Swin 必须工作在论文空间网格语义下，不再依赖 HEALPix face reshape。

可选 [REPO]：

- regular lat-lon 2D windows
- longitude periodic wrap
- latitude non-periodic padding/masking

但这仍不是 paper-exact。

---

## 8.3 SFNO modification 2 — Spectral filter compression

原 SFNO 每个 spherical harmonic degree \(l\) 有独立 filter：

\[
\tilde{\kappa}_{\vartheta}(l).
\]

Huracan：

1. sinusoidal encoding of \(l\)
2. concatenate noise embedding
3. MLP:
   \[
   h(l)\in\mathbb R^k
   \]
4. construct filter:

\[
\boxed{
\kappa(l)=\sum_{i=1}^{k}h_i(l)\tilde{\kappa}_i
}
\]

**[PAPER]**

\[
L=179,\qquad k=15.
\]

论文称参数量约降低 12×。

### 注意

论文公式没有：

\[
\operatorname{softmax}(h)
\]

所以默认不应人为增加归一化约束。

---

## 8.4 SFNO modification 3 — stochastic conditioning

**[PAPER]**

\[
\epsilon\sim\mathcal N
\]

经过 MLP processor 产生 noise embeddings。

这些 embeddings：

- 是每 grid point 的 context；
- 输入 conditional LayerNorm；
- 替换标准 LayerNorm；
- 用于产生 stochastic ensemble。

### [OPEN] noise semantics

论文没有明确：

- Gaussian noise 的维度
- global / per-grid
- independent per variable?
- MLP depth / hidden dim
- 是否每 block 独立 processor
- 是否同一 member 整个 trajectory 固定 noise
- 是否每个 model call resample
- assimilation / forecast noise 是否共享 stream

这些会显著影响 temporal ensemble coherence。

---

## 8.5 双时步 IO

**[PAPER]**

输入：

\[
(T_0-6h,T_0)
\]

输出：

\[
(T_0+6h,T_0+12h)
\]

time dimension internally flattened into channels。

### 推荐 tensor layout

例如：

```text
input_state:
[B, 2, C_state, H, W]

flattened:
[B, 2*C_state (+ forcing + obs), H, W]

output:
[B, 2, 72, H, W]
```

实际 encoder flatten 方式属于 [REPO]，但必须保证时间语义一致。

---

## 8.6 Residual prediction 不属于 paper fact

旧版曾采用：

\[
\hat x_{future}=x_t+\Delta x.
\]

并 zero-initialize output layer。

论文没有说明：

- residual target
- delta prediction
- output zero initialization

因此：

> **[REPO] 可以继续使用 residual prediction 作为稳定训练选择，但不得标成 Sec. 2.2 的论文设计。**

建议做 ablation：

```text
direct-state prediction
vs
residual prediction
```

---

# 9. 训练流程

## 9.1 总体依赖

```text
deterministic base
       │
       ▼
stochastic base
    ┌──┴───────────────┐
    ▼                  ▼
forecast AR       assimilation AR
    └──────────┬───────┘
               ▼
        final joint Huracan

optional:
forecast AR -> ECMWF HRES fine-tune -> Huracan-Hybrid
```

---

## 9.2 全局训练参数

**[PAPER]**

- batch size = 16
- stochastic training ensemble size = 2

---

## 9.3 Phase 1 — deterministic base

| 项 | 值 |
|---|---|
| data | ERA5 1979–2014 |
| loss | L1 |
| max LR | \(5\times10^{-4}\) |
| schedule | cosine |
| steps | 347k |
| stochastic noise | no / deterministic |

---

## 9.4 Phase 2 — stochastic base

| 项 | 值 |
|---|---|
| data | ERA5 1979–2014 |
| objective | CRPS |
| max LR | \(2\times10^{-4}\) |
| schedule | cosine |
| steps | 149k |
| training ensemble | 2 |

此阶段引入 stochastic noise conditioning。

---

## 9.5 Phase 3 — initial forecast model

| 项 | 值 |
|---|---|
| data | ERA5 1979–2014 |
| max LR | \(3\times10^{-6}\) |
| schedule | constant |
| steps | 14k |
| rollout | 6 autoregressive calls |
| rollout horizon | 3 days |

因为每 call 推进 12 h：

\[
6\times12h=72h.
\]

---

## 9.6 Phase 4 — initial assimilation model

**初始化**

从 stochastic base model 继承所有可匹配权重。

只有：

> encoder 中新增 observation-channel columns 随机初始化。

**训练**

| 项 | 值 |
|---|---|
| data | ERA5 + Obs, 2010–2022 |
| max LR | \(2\times10^{-4}\) |
| schedule | cosine |
| steps | 46k |
| rollout | 8 autoregressive calls |
| horizon | 4 days |

论文还说明：

- initial condition 来自 ERA5；
- 每步 ground truth 来自 ERA5；
- observations 来自观测 archive。

### 设计动机

4-day long rollout 迫使 assimilation model 利用 observation，而不是长期依赖 ERA5 background。

---

## 9.7 Phase 5 — final joint Huracan

两个模型一起 fine-tune：

\[
8\text{ assimilation calls}
+
8\text{ forecast calls}.
\]

| 项 | 值 |
|---|---|
| data | ERA5 + Obs, 2010–2022 |
| max LR | \(3\times10^{-6}\) |
| schedule | cosine |
| steps | 13k |

论文没有说明：

- assimilation loss / forecast loss 的相对权重
- rollout-step weighting
- 是否所有 32 个 6 h output fields 等权
- backprop truncation
- detach policy

这些属于 [OPEN]。

---

## 9.8 Phase 6 — Huracan-Hybrid

**可选 baseline，不属于 end-to-end Huracan 本体。**

| 项 | 值 |
|---|---|
| base | initial forecasting model |
| fine-tune data | ECMWF HRES 2018–2022 |
| max LR | \(3\times10^{-6}\) |
| schedule | constant |
| steps | 2.7k |
| inference IC | ECMWF ENS ensemble initial conditions |

---

# 10. CRPS 训练损失

论文只说明使用 CRPS，以及 stochastic training ensemble size = 2。

**[OPEN]**

- biased empirical CRPS or fair CRPS
- latitude weighting
- variable weighting
- normalization before/after CRPS
- vertical-level weighting

### 推荐 [REPO] fair ensemble CRPS

对 ensemble \(\{x_m\}_{m=1}^M\)：

\[
\mathrm{CRPS}_{fair}
=
\frac1M\sum_{m=1}^{M}|x_m-y|
-
\frac{1}{M(M-1)}
\sum_{m<m'}|x_m-x_{m'}|.
\]

对于 \(M=2\)：

\[
\mathrm{CRPS}_{fair}
=
\frac{|x_1-y|+|x_2-y|}{2}
-\frac12|x_1-x_2|.
\]

但必须在代码与报告中写：

> `loss_variant = repo_choice_fair_crps`

而不是 “Huracan uses fair CRPS”。

---

# 11. Optimizer 与 schedule

论文只公布：

- maximum LR
- cosine / constant
- step count

以下均 [OPEN]：

- optimizer
- Adam/AdamW betas
- weight decay
- epsilon
- warmup
- min LR
- gradient clipping
- mixed precision
- gradient accumulation
- EMA
- checkpoint policy

### 推荐 [REPO]

```yaml
optimizer:
  name: AdamW
  weight_decay: 0.01
  grad_clip_norm: 1.0

schedule:
  warmup_fraction: 0.02
  cosine_min_lr: 0.0
```

这些只作为工程默认值。

---

# 12. EDA 推理

## 12.1 初始化

**[PAPER]**

48 个 independent EDA members。

所有成员从同一个初值开始：

```text
ERA5
2023-10-01 00 UTC
```

之后通过 stochastic model 独立发展。

---

## 12.2 正确的 member propagation

成员不能每 cycle 被重置到 ensemble mean。

概念上：

\[
(x_{t-6}^{(m)},x_t^{(m)},y)
\rightarrow
(x_{t+6}^{(m)},x_{t+12}^{(m)}).
\]

下一 call：

\[
(x_{t+6}^{(m)},x_{t+12}^{(m)},y')
\rightarrow
(x_{t+18}^{(m)},x_{t+24}^{(m)}).
\]

所以 spread 会同时来自：

1. 历史 stochastic perturbation；
2. 当前 model noise；
3. 非线性 propagation。

---

## 12.3 Spin-up 修订

论文：

> at least three months of data assimilation

旧版：

```text
720 cycles = 3 months
```

是错误的，因为旧版误把一个 AR call 当成 6 h。

若按 12 h advancement：

```text
2023-10-01 00 UTC
→ 2024-01-01 00 UTC
≈ 92 days
≈ 184 autoregressive calls
```

但精确值应由 calendar timestamps 驱动，而不是写死循环数。

### 推荐实现

```python
while current_time < evaluation_start:
    run_one_12h_assimilation_call()
```

而不是：

```python
for _ in range(720):
    ...
```

### [OPEN]

论文没有公开：

- spin-up 到底取 3 个月整还是更长；
- 2024 每个 issue time 是从全年连续 EDA stream 截取，还是存在其他 reset 策略。

最自然的实现是：

> 从 2023-10-01 起持续运行 48-member EDA，全年不 reset。

---

# 13. Forecast inference

模型每 call 生成：

\[
+6h,+12h
\]

两个输出。

10-day forecast：

\[
240h / 12h = 20
\]

个 autoregressive model calls。

但输出 forecast fields 为：

\[
40
\]

个：

```text
+6h, +12h, ..., +240h
```

如果再包括 initialization / analysis 的 lead 0：

```text
0, 6, 12, ..., 240 h
```

总计：

\[
41
\]

个 lead positions。

---

# 14. 697 个 scorecard combination 的一致性检查

论文评分变量：

- z × 4 levels
- t × 4 levels
- q × 4 levels
- 5 surface vars

所以：

\[
3\times4+5=17.
\]

论文报告：

\[
697
\]

个 variable × lead-time combinations。

而：

\[
697/17=41.
\]

恰好：

\[
0h,6h,\dots,240h
\]

共 41 个 lead positions。

因此可以作为非常强的 consistency check：

> **[INFERRED] scorecard 很可能包含 lead-0 analysis + 40 个 6-hourly forecast leads。**

实现评估脚本时应支持这一布局，但仍需把“lead-0 是否进入论文 scorecard”标记为 inference，直到官方代码确认。

---

# 15. 评估协议

## 15.1 Issue times

**[PAPER]**

2024 年：

- 100 个 issue times
- 范围：Jan 1 – Dec 17
- 选择受 ECMWF hindcast availability 约束

**[OPEN]**

论文没有列出全部 100 个时间戳。

因此无法仅凭论文实现 exact issue-time reproduction。

---

## 15.2 Forecast horizon

- 10 lead days
- output spacing = 6 h

---

## 15.3 Analysis / truth

### Huracan

\[
x^{analysis}
=
\frac{1}{48}\sum_{m=1}^{48}x^{(m)}_{EDA}.
\]

### ECMWF ENS

使用 ECMWF 自己 analysis。

论文比较是：

> each system evaluated against its own analysis.

因此不同系统的 truth 并不完全相同。

### pilot 注意

若 pilot 全部对 ERA5：

> 只能验证 pipeline 和相对 behavior，不能声称复现论文 scorecard。

---

## 15.4 ECMWF resolution matching

**[PAPER]**

ECMWF：

- initial conditions
- forecasts
- analyses

全部 conservative downsample 到 1°。

---

## 15.5 评分变量

高空：

- z @ 200, 500, 700, 850 hPa
- t @ 200, 500, 700, 850 hPa
- q @ 200, 500, 700, 850 hPa

地面：

- 2t
- 2d
- 10u
- 10v
- msl

总数：

\[
17.
\]

`tcc` 与 `tcwv` 不进入论文主要 scorecard。

---

# 16. 评估指标

## 16.1 CRPS

论文明确使用 CRPS。

但没有给出完整 numerical implementation：

- finite-ensemble estimator
- area weighting
- handling of poles
- spatial average order

均 [OPEN]。

---

## 16.2 CRPS skill score

论文称沿用 GraphCast notation，以 relative differences 展示。

复现时建议单独实现并写单元测试，明确 sign convention，例如：

\[
\mathrm{skill}
=
\frac{\mathrm{CRPS}_{baseline}-\mathrm{CRPS}_{model}}
{\mathrm{CRPS}_{baseline}}.
\]

若论文图的颜色约定不同，必须按 figure 方向校验。

---

## 16.3 ESSR

论文使用 ensemble spread-skill ratio。

但未给完整公式，因此需显式记录是否使用 finite-ensemble correction。

可能形式：

\[
\mathrm{ESSR}
=
\frac{\mathrm{Spread}}{\mathrm{RMSE}(\bar x)}
\]

或：

\[
\sqrt{\frac{M+1}{M}}
\frac{\mathrm{Spread}}{\mathrm{RMSE}(\bar x)}.
\]

当前属于 [OPEN]。

---

# 17. 论文报告的行为特征

这些是很好的 reproduction sanity checks。

## 17.1 初期 ESSR 较高

Huracan：

- 没有 deterministic control member；
- 48 members independent；
- analysis = ensemble mean。

论文认为因此 Huracan analysis 更接近 expectation，初期 ESSR 会偏高。

---

## 17.2 3–5 天 spread 偏低

论文认为可能与 overfitting 有关。

训练年 2017 上：

> ESSR reportedly remains above 1 throughout 10 days.

---

## 17.3 train-test gap

论文明确承认：

- training year performance 明显优于 2024 test year。

所以正式复现应至少报告：

```text
one training-year diagnostic
2023 validation
2024 test
```

---

## 17.4 ECMWF 未来观测公平性

论文讨论指出：

ECMWF 00 UTC IC 实际会使用到约 03 UTC 才收到的观测，而 Huracan 不使用这种 future observations。

这是比较时需要注明的 methodological difference。

---

# 18. 完整 Open Questions / 不可恢复项

以下清单是论文级 faithful reproduction 的核心。

## A. Grid / SHT

| ID | 未公开项 | 风险 |
|---|---|---|
| G01 | final 1° grid 的精确 latitude nodes | HIGH |
| G02 | longitude origin / ordering | MED |
| G03 | 是否 180×360 或 181×360 | HIGH |
| G04 | poles 是否包含 | HIGH |
| G05 | SHT grid convention / quadrature | HIGH |
| G06 | spectral order \(m\) truncation | MED |
| G07 | SHT normalization | MED |
| G08 | ERA5 → final grid 的精确 remap | HIGH |

## B. Model architecture

| ID | 未公开项 | 风险 |
|---|---|---|
| M01 | hidden dimension | HIGH |
| M02 | number of SFNO blocks | HIGH |
| M03 | encoder architecture | HIGH |
| M04 | decoder/output head | HIGH |
| M05 | activation | MED |
| M06 | norm placement / pre/post norm | HIGH |
| M07 | Swin heads | HIGH |
| M08 | Swin window | HIGH |
| M09 | shift pattern | MED |
| M10 | MLP ratio | MED |
| M11 | dropout / drop-path | MED |
| M12 | exact modified SFNO block topology beyond Fig. 2 | HIGH |
| M13 | residual-vs-direct target | HIGH |
| M14 | output-layer initialization | MED |
| M15 | parameter count | LOW |

## C. Stochasticity

| ID | 未公开项 | 风险 |
|---|---|---|
| S01 | Gaussian noise dimension | HIGH |
| S02 | global vs per-grid noise | HIGH |
| S03 | noise MLP architecture | HIGH |
| S04 | context embedding dimension | HIGH |
| S05 | noise sharing across blocks | MED |
| S06 | resample per AR call or fixed per member | HIGH |
| S07 | assimilation/forecast noise-stream relationship | HIGH |
| S08 | random seeds / reproducibility policy | LOW |

## D. Pressure levels / state

| ID | 未公开项 | 风险 |
|---|---|---|
| V01 | exact 13 pressure levels | HIGH |
| V02 | physical-variable normalization | HIGH |
| V03 | variable weighting in losses | HIGH |
| V04 | latitude/area weighting in losses | HIGH |

## E. Observations

| ID | 未公开项 | 风险 |
|---|---|---|
| O01 | 269 vs 334 exact encoder channel count | CRITICAL |
| O02 | exact cross-instrument channel matching table | HIGH |
| O03 | product-specific QC flags | HIGH |
| O04 | higher-resolution intermediate grid | HIGH |
| O05 | conservative remap implementation | MED |
| O06 | exact temporal frame boundaries | HIGH |
| O07 | tie breaking | LOW |
| O08 | multi-orbit overlap reduction | MED |
| O09 | multi-satellite merge order | MED |
| O10 | observation normalization statistics | HIGH |
| O11 | missing-zero ambiguity / masks | MED |
| O12 | metadata normalization | MED |

## F. Autoencoders

| ID | 未公开项 | 风险 |
|---|---|---|
| A01 | AE hidden widths | HIGH |
| A02 | activation | MED |
| A03 | normalization | MED |
| A04 | optimizer / LR | MED |
| A05 | batch / steps | MED |
| A06 | training sampling | HIGH |
| A07 | spectral normalization | HIGH |
| A08 | exact six RO features | CRITICAL |
| A09 | CrIS apodization exact algorithm | HIGH |
| A10 | IR interpolation kernel / boundary behavior | MED |

## G. Training

| ID | 未公开项 | 风险 |
|---|---|---|
| T01 | optimizer | HIGH |
| T02 | weight decay | MED |
| T03 | warmup | MED |
| T04 | cosine minimum LR | MED |
| T05 | gradient clipping | LOW/MED |
| T06 | precision | LOW |
| T07 | rollout loss aggregation | HIGH |
| T08 | joint DA-vs-forecast loss weight | HIGH |
| T09 | BPTT/detach policy | HIGH |
| T10 | new conditional-LN initialization when entering stochastic phase | HIGH |
| T11 | checkpoint selection criterion | MED |

## H. EDA / inference

| ID | 未公开项 | 风险 |
|---|---|---|
| I01 | exact spin-up end/start policy | MED |
| I02 | continuous yearly EDA vs issue-time resets | HIGH |
| I03 | noise resampling semantics during EDA | HIGH |
| I04 | exact handling of observation windows at 6 h intermediate output | HIGH |
| I05 | forecast branch stochastic seed relationship to analysis member | MED |

## I. Evaluation

| ID | 未公开项 | 风险 |
|---|---|---|
| E01 | exact 100 issue timestamps | HIGH |
| E02 | CRPS estimator | MED |
| E03 | area weighting | HIGH |
| E04 | ESSR definition | MED |
| E05 | finite-ensemble correction | MED |
| E06 | exact scorecard aggregation | MED |
| E07 | lead-0 inclusion | LOW/MED，697 强烈支持 inclusion |
| E08 | “first two noisy steps” 在统计表中的具体排除规则 | MED |

---

# 19. 推荐配置分层

不要再使用一个看起来像“论文真值”的 `paper.yaml`。

推荐：

```text
configs/
├── paper_constraints.yaml
├── scaleup_inferred.yaml
└── pilot.yaml
```

## 19.1 `paper_constraints.yaml`

只放论文明确值：

```yaml
resolution_deg: 1.0

model:
  state_channels: 72
  input_times: 2
  output_times: 2
  output_interval_hours: 6
  ar_call_advance_hours: 12

spectral:
  lmax: 179
  k_filters: 15

training:
  batch_size: 16
  stochastic_ensemble_size: 2

rollout:
  forecast_calls: 6
  assimilation_calls: 8
  joint_assimilation_calls: 8
  joint_forecast_calls: 8

eda:
  members: 48
  initial_time: "2023-10-01T00:00:00Z"
  min_spinup_months: 3

evaluation:
  forecast_days: 10
  test_year: 2024
  issue_times: 100
```

## 19.2 `scaleup_inferred.yaml`

加入：

```yaml
grid:
  family: latlon
  sht_grid: equiangular
  nlat: 180
  nlon: 360

model:
  hidden_dim: 512
  num_blocks: 12
  num_heads: 8
  swin_window: 8
  mlp_ratio: 4.0

noise:
  noise_dim: 16
  embed_dim: 128

optimizer:
  name: AdamW
  weight_decay: 0.01
```

所有这些字段旁应注释：

```text
# REPO / not specified by paper
```

---

# 20. 仓库架构建议（修订）

```text
huracan/
├── config/
│   ├── schema.py
│   └── loader.py
├── core/
│   ├── grid.py
│   ├── sht.py
│   ├── spectral.py
│   ├── swin_latlon.py
│   ├── modules.py
│   ├── model.py
│   ├── rollout.py
│   ├── losses.py
│   └── metrics.py
├── data/
│   ├── era5.py
│   ├── observations/
│   │   ├── microwave.py
│   │   ├── hyperspectral.py
│   │   ├── ro.py
│   │   ├── geo.py
│   │   └── insitu.py
│   ├── remap.py
│   ├── temporal_frames.py
│   ├── channel_spec.py
│   └── normalization.py
├── training/
│   ├── phases.py
│   ├── rollout_loss.py
│   └── schedules.py
├── inference/
│   ├── eda.py
│   └── forecast.py
└── eval/
    ├── crps.py
    ├── essr.py
    └── scorecard.py
```

### 必须删除/重构的旧模块语义

若旧仓库存在：

```text
HEALPix
nside
NESTED
SwinFaceBlock
face wrap
MatrixSHT for HEALPix
```

应从 faithful Huracan path 移除。

如果这些模块还想保留实验，可以放：

```text
experiments/healpix_variant/
```

明确标成 variant。

---

# 21. 单元测试要求

## 21.1 时间语义

```python
input_times = [-6, 0]
output_times = [6, 12]

second_call_input = [6, 12]
second_call_output = [18, 24]
```

必须 assert：

```text
6 calls -> 72 h
8 calls -> 96 h
20 calls -> 240 h
```

---

## 21.2 Channel accounting

程序输出：

```text
model state             72
forcing/constants        3
obs table sum          334
obs text claim         269
difference              65
```

不得 silent truncate。

---

## 21.3 SHT

在目标 grid 上测试：

- forward/inverse consistency
- normalization
- lmax=179
- FP32 stability

如果使用 AMP：

> SHT 区域建议强制 FP32，避免 FFT/SHT 对半精度 shape 的限制与数值问题。

---

## 21.4 Rollout

必须单测：

- forecast rollout
- assimilation rollout
- joint rollout
- member independence
- no reset to ensemble mean

---

## 21.5 Missing observations

需要测试：

```text
after normalization:
missing -> exact zero
```

并验证有效观测值不会被数据 loader 的 mask 逻辑误当成 missing。

---

# 22. Pilot 复现路线

## Level 0 — synthetic smoke test

目标：

- architecture forward
- 2-time IO
- stochastic two-member CRPS
- 6/8-step rollout
- EDA member propagation
- evaluation API

不验证论文精度。

---

## Level 1 — reduced real-data pilot

建议：

- regular lat-lon coarse grid
- ERA5 subset
- IGRA + ISD
- no satellite radiance initially
- 8–16 ensemble members
- shortened spin-up

### 注意

pilot 可以缩：

```text
grid resolution
hidden dim
num blocks
training steps
ensemble size
years
```

但**不能缩错时间语义**。

即使 coarse pilot：

```text
one AR call = 12h
two outputs/call
```

也必须保留。

---

## Level 2 — observation-complete pilot

加入：

- microwave
- geostationary
- hyperspectral AE
- RO AE
- metadata channels
- temporal framing
- remapping

重点先验证：

1. channel accounting；
2. missing-data behavior；
3. observation-only ablations；
4. assimilation actually uses observations。

---

## Level 3 — paper-scale training

需要：

- 1979–2014 ERA5
- 2010–2022 observation archive
- distributed training
- torch-harmonics / scalable SHT
- full six training phases
- 48-member EDA
- continuous spin-up
- 2024 10-day forecasts

---

# 23. Scale-up 前必须解决的 P0/P1 问题

## P0 — 不解决就不要跑论文规模

- [ ] 删除 HEALPix faithful-path
- [ ] regular spherical grid 实现
- [ ] 12 h autoregressive call 语义完全修正
- [ ] calendar-driven spin-up
- [ ] 269/334 channel accounting 不 silent fix
- [ ] exact member propagation
- [ ] residual prediction 从 paper fact 降为 repo choice
- [ ] all config fields provenance tagging

## P1 — full-observation training 前解决

- [ ] IR preprocessing
- [ ] full RO features strategy
- [ ] channel frequency matching table
- [ ] product-specific QC
- [ ] temporal frame implementation
- [ ] observation normalization
- [ ] high-resolution intermediate remap
- [ ] AE training recipe

## P2 — exact paper-score comparison 前解决

- [ ] exact 100 issue times
- [ ] ECMWF HRES / ENS / analysis
- [ ] exact CRPS aggregation
- [ ] ESSR definition
- [ ] scorecard lead convention
- [ ] common remapping validation

---

# 24. 建议的实验矩阵

为了判断实现是否真的学到了 assimilation，而不是只靠 background：

| 实验 | Obs | stochastic | long rollout | 目的 |
|---|---|---|---|---|
| deterministic fcst | no | no | no | base |
| stochastic fcst | no | yes | no | ensemble mechanism |
| short DA | yes | yes | short | observation ingestion |
| long DA | yes | yes | 8 calls | test background decay |
| shuffled Obs | shuffled | yes | 8 calls | causal sanity check |
| zero Obs | zero | yes | 8 calls | quantify observation value |
| full Huracan | yes | yes | joint | target |

特别建议：

\[
\Delta L
=
L_{\text{zero obs}}-L_{\text{real obs}}
\]

作为 assimilation-use sanity metric。

---

# 25. 数据源获取速查

| 数据 | 提供方 | 获取方式 / 典型平台 | 门槛 |
|---|---|---|---|
| AMSU-A/B, MHS | NOAA CLASS | TOVS family | 免费注册 |
| ATMS SDR | NOAA CLASS / NASA | ATMS products | 免费注册 |
| CrIS / AIRS | NASA GES DISC | L1 products | 免费注册 |
| IASI / Metop microwave | EUMETSAT | Data Store / Navigator | 注册/许可 |
| GNSS-RO | UCAR / AWS Open Data | RO archives | 开放 |
| GOES ABI | NOAA AWS Open Data | GOES archive | 开放 |
| Himawari AHI | JMA/NOAA AWS | Himawari archive | 开放 |
| GridSat | NOAA NCEI | GridSat archives | 开放 |
| IGRA | NOAA NCEI | IGRA v2 | 开放 |
| ISD | NOAA NCEI | ISD | 开放 |
| ERA5 | Copernicus CDS | CDS / derived public mirrors | 注册或公开镜像 |
| ECMWF HRES / ENS | ECMWF | MARS 等 | 通常需许可 |

---

# 26. 推荐 provenance 机制

每个关键超参最好保存来源。

例如：

```yaml
model:
  hidden_dim:
    value: 512
    source: repo_choice

spectral:
  lmax:
    value: 179
    source: paper_appendix_A2

training:
  batch_size:
    value: 16
    source: paper_appendix_A3
```

或维护一个自动生成表：

| config key | value | provenance |
|---|---:|---|
| `spectral.lmax` | 179 | PAPER A.2 |
| `spectral.k_filters` | 15 | PAPER A.2 |
| `model.hidden` | 512 | REPO |
| `grid.nlat` | 180 | INFERRED/REPO |
| `optimizer.name` | AdamW | REPO |

这样后续官方代码发布时可以快速 diff。

---

# 27. 对当前复现可行性的最终判断

## 可以高置信复现的部分

- 系统两模型结构关系
- observation categories
- 大部分 preprocessing high-level pipeline
- 72 state variables 的变量种类
- 2-time input / 2-time output
- SFNO 三项核心修改
- spectral filter compression formula
- \(L=179,k=15\)
- six-stage training schedule
- steps / max LR / schedule
- batch=16 / stochastic M=2
- 6-call / 8-call rollout horizons
- 48-member EDA
- initialization date
- ≥3-month spin-up principle
- 2024 evaluation framework
- scorecard variables

## 不能仅凭论文精确复现的部分

- exact grid nodes
- exact 13 pressure levels
- model width/depth
- Swin hyperparameters
- observation channel accounting
- detailed sensor QC
- channel matching map
- AE hidden architecture
- full six RO features
- optimizer recipe
- stochastic-noise semantics
- loss weighting
- exact issue times
- exact metric aggregation

因此正确定位应是：

> **A faithful paper-constrained reimplementation with explicit unresolved choices**,  
> 而不是“official reproduction”。

---

# 28. 与旧版指南相比的修订记录

### Critical fixes

1. **删除 HEALPix/nside/NESTED 作为 Huracan 原始网格的假设。**
2. 改为 regular lat-lon / equiangular SFNO 高置信路线。
3. **修正 autoregressive step：1 call = 12 h advancement。**
4. `6 calls = 3 days`、`8 calls = 4 days`。
5. 10-day forecast = 20 calls = 40 future 6 h fields。
6. 修正 spin-up，不再使用 `720 × 6h`。
7. spin-up 改为 calendar-driven；Oct 1 → Jan 1 约 184 个 12 h calls。
8. residual prediction / zero-init 降为 [REPO]。
9. `paper.yaml` 建议改为 `paper_constraints.yaml + scaleup_inferred.yaml`。
10. 269/334 不再通过配置强行“对齐 269”。

### Completeness additions

新增并明确：

- exact grid ambiguity
- SHT convention
- pressure-level risk
- observation QC gaps
- channel matching gap
- temporal frame ambiguity
- normalization ambiguity
- AE architecture gaps
- exact RO features gap
- noise temporal semantics
- rollout/joint-loss aggregation
- exact issue-time gap
- ESSR/CRPS weighting gap
- 697 = 17 × 41 consistency check

---

# 29. 最终复现 checklist

## 数据

- [ ] ERA5 1979–2014
- [ ] ERA5 2010–2022
- [ ] 2023 validation
- [ ] 2024 test
- [ ] full observation archive
- [ ] ECMWF HRES/ENS if reproducing hybrid

## 网格

- [ ] no HEALPix assumption
- [ ] regular SHT-compatible grid
- [ ] lmax=179
- [ ] conservative remap verified
- [ ] latitude/longitude convention recorded

## Observation

- [ ] QC
- [ ] spatial interpolation
- [ ] temporal frames
- [ ] matching frequencies
- [ ] metadata channels
- [ ] normalization
- [ ] missing=0
- [ ] IR AE
- [ ] RO AE
- [ ] 269/334 accounting printed

## 模型

- [ ] SFNO
- [ ] Swin replacement
- [ ] compressed spectral filters
- [ ] Gaussian-noise context
- [ ] conditional LN
- [ ] dual-time IO
- [ ] repo-only architectural choices documented

## 训练

- [ ] deterministic 347k
- [ ] stochastic 149k
- [ ] forecast AR 14k / 6 calls
- [ ] assimilation 46k / 8 calls
- [ ] joint 13k / 8+8
- [ ] hybrid 2.7k optional
- [ ] batch=16
- [ ] train ensemble=2

## EDA

- [ ] 48 members
- [ ] same initial ERA5 state
- [ ] independent trajectories
- [ ] no mean reset
- [ ] continuous assimilation
- [ ] ≥3 months spin-up
- [ ] calendar-driven time

## Forecast

- [ ] 20 calls for 10 days
- [ ] outputs at every 6 h
- [ ] +240 h included

## Evaluation

- [ ] Huracan analysis = 48-member EDA mean
- [ ] ECMWF own analysis
- [ ] 17 evaluation variables
- [ ] CRPS
- [ ] CRPS skill score
- [ ] ESSR
- [ ] exact issue-time list documented when available
- [ ] area weighting documented
- [ ] 697-combination consistency check

---

# Appendix A. 变量速查

## Upper-air

\[
z,t,u,v,q
\]

× 13 pressure levels.

## Surface

```text
2t
2d
10u
10v
tcc
tcwv
msl
```

## Scorecard

```text
z200 z500 z700 z850
t200 t500 t700 t850
q200 q500 q700 q850
2t 2d 10u 10v msl
```

---

# Appendix B. 时间语义示例

初始：

```text
input:
t=-6h
t= 0h
```

Call 1：

```text
output:
t=+6h
t=+12h
```

Call 2：

```text
input:
t=+6h
t=+12h

output:
t=+18h
t=+24h
```

所以：

```text
AR call 1 -> 12 h horizon
AR call 2 -> 24 h
AR call 6 -> 72 h = 3 d
AR call 8 -> 96 h = 4 d
AR call 20 -> 240 h = 10 d
```

---

# Appendix C. 必须避免的错误表述

不要写：

> Huracan uses HEALPix nside=64.

应写：

> The paper specifies 1° SFNO and \(L=179\), but does not specify HEALPix.  
> We implement an SHT-compatible regular lat-lon/equiangular grid as a high-confidence inference.

不要写：

> Huracan runs one 6-hour autoregressive step.

应写：

> Huracan outputs two 6-hourly states per model call; paper rollout lengths imply one autoregressive call advances 12 hours.

不要写：

> Huracan uses residual prediction.

应写：

> Our implementation optionally uses residual prediction; the paper does not specify the target parameterization.

不要写：

> Huracan has exactly 269 observation channels according to Table 2.

应写：

> The main text reports 269, while Table 2 entries sum to 334; the discrepancy is unresolved.

不要写：

> paper.yaml exactly reproduces the official architecture.

应写：

> `scaleup_inferred.yaml` satisfies known paper constraints but contains explicit implementation assumptions for unpublished hyperparameters.

---

# Appendix D. Primary reference

Ni et al., **Huracan: A Skillful End-To-End Data-Driven System for Ensemble Data Assimilation and Weather Prediction**, arXiv:2508.18486v2.

主要核对位置：

- Sec. 2.1 — Data and observations
- Table 1 — Observation sources
- Table 2 — Model variables / observation channels
- Sec. 2.2 — Architecture
- Sec. 2.3 — Training
- Sec. 3.1 — Forecast skill / EDA initialization / evaluation
- Sec. 4 — Discussion
- Appendix A.1 — IR/RO embeddings
- Appendix A.2 — SFNO filter compression
- Appendix A.3 / Table 3 — Training parameters

---

*Revision note: 本版本用于替换旧 `REPRODUCTION_GUIDE.md`。如果未来作者发布官方代码或补充材料，应优先以官方实现更新所有 `[OPEN]` 与 `[INFERRED]` 项，并保留 provenance diff。*
