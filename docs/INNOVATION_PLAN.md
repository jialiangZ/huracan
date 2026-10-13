# 创新方案：观测 token 化 + 流匹配集合生成（INNOVATION PLAN v1.0）

> 2026-10-12 定稿。两项方法学创新（A：站点观测 token 化；B：流匹配
> 集合生成），全部落在现有三对照 OSSE 协议（with-obs / zero-obs /
> fcst-spinup）内验证，"with-obs 分析误差随循环下降"仍是第一判据。
> 本方案遵循仓库既有原则：先改 schema 再加 yaml；单一权威实现；
> 数据契约统一（state/obs/forcing/norms + 点观测扩展）。

---

## 0. 决策记录（已确认，不再讨论）

| # | 决策项 | 选择 | 主要影响 |
|---|--------|------|----------|
| 1 | 创新方向 | **A 观测 token 化 × B 流匹配集合**，两条线独立可消融、最终组合 | 接触面仅在 `step()` 签名，可并行推进 |
| 2 | token 注入深度 | **逐 block cross-attention**（表达力最强） | 实现 `inject_every` 参数（1=逐 block，N=每 N 个 block 注一次），保留入口层/逐 block 消融阶梯，不必二选一 |
| 3 | 扩散起点 | **B1 FM head 先行**（确定性主干 + 流匹配随机增量头），验证后升级 B2 full FM trunk | deterministic phase 及 checkpoint 零改动，首个可跑结果 1–2 周 |
| 4 | 位置编码 | 球面：**实球谐基** Y_lm 到 lmax_pos≈16（与 sht.py 同族）；L96：1D sinusoidal | 几何原生；`spherical_harmonics_basis(lat, lon, lmax)` 工具 + 与网格基互验单测 |
| 5 | 点观测契约 | `obs_points.npz`（values/mask/连续站位/channel_ids）+ gridded obs.npy **双渲染，一份真相** | A/B 公平：baseline 与 token 模型消费同一组站点数据 |
| 6 | 先决修复 | **obs 缺失 mask**（现状缺失=精确 0，与气候均值不可区分） | 公平 baseline 的前提；grid 路径加 presence 通道 |
| 7 | 采样成本 | FM 训练 rollout 与 EDA 循环均用 **1-step Euler**；评估时扫步数 {1,2,4,8,16} | 训练/推理成本与现有 noise-cond 同阶，步数 sweep 是独立成果 |
| 8 | 参数公平性 | token 分支 / FM head 参数计入总预算，与 baseline 对齐（±10% 内） | 消融结论可信的前提 |

---

## 1. 现状事实（代码确认，设计依据）

1. **观测是像素级不是站点级**：`turbulence.py::write_observations` 里 OSSE
   以像素为站点（`observation_network` 返回 pixel indices）；`data/obs.py`
   （IGRA/ISD）贴最近像素、同像素多站平均。亚网格位置信息全管线不存在，
   "离网格观测"场景目前无法表达。
2. **缺失 = 精确 0，无 mask**：`obs[~active] = 0.0`；归一化后 0 = 气候
   均值，模型原则上无法区分"未观测"与"观测值恰为均值"。
3. **集合生成是噪声条件化的确定性映射**：`z (B,npix,noise_dim)` →
   `NoiseEmbedder` → 逐像素 `e`（条件化全部 ConditionalLayerNorm）+ 池化
   `g`（条件化 `CompressedSpectralFilter` 度数权重 h(l)）。成员差异仅能
   经条件 LN / 谱滤波权重进入，生成容量低，spread 标定完全靠 CRPS 间接学。

---

## 2. A 线：站点观测 token 化

### 2.1 数据契约（点观测格式）

```
obs_points.npz:
  values:        (T, S, C)  float16  归一化观测值（norm_obs 同 gridded 路径）
  mask:          (T, S)     bool     报数掩码（obs_report_prob < 1 时部分 False）
  station_lat:   (S,)       float32  连续位置（不对齐像素！）
  station_lon:   (S,)       float32
  channel_ids:   (C,)       int      → 状态通道索引（obs 通道→state 通道映射）
```

- **双渲染原则**：`write_observations` 重构为"采样连续站点位置 → 同时
  渲染 gridded obs.npy（贴像素平均，喂 baseline grid 路径）与
  obs_points.npz（喂 token 路径）"。`make_obs_variant` 的密度/噪声 sweep
  改造为连续站位重采样（symlink 重逻辑不变）。
- IGRA/ISD 线：`data/obs.py` 在贴网格前本有逐站数据，加一个保存点；
  站位 = 站点真实经纬度（天然离网格，即实验 E5）。
- L96 线：站位 = 变量索引 k（1D），`station_lat/lon` 退化为 `station_idx`。

### 2.2 模型：ObsTokenEncoder + 逐 block 注入

**共享 token 嵌入（编码一次，逐 block 复用）**：

```
token_feat = MLP([ value_emb(C→d) , channel_emb(可学习表) ,
                   pos_emb(SH 基/1D sinusoidal) , frame_emb(T-6h / T 两帧) ])
输出 (B, 2S, d_tok)（两帧 token 拼接，S 为站点数）
```

**block 内第三残差分支**（`HuracanSFNOBlock` / `Hurac1DBlock` 同步）：

```
h  = spec_norm(x, e);  x = x + ISHT(filter(SHT(h), g))     # 现有谱分支
x  = x + mixer(mixer_norm(x, e))                            # 现有 Swin 分支
x  = x + obs_attn(attn_norm(x, e), tokens)   # ← 新增：cross-attention
```

- `obs_attn`：Q = 逐像素 hidden（块内独立 Q 投影）；K/V = token_feat
  （块内独立投影，从 d_tok → hidden）；多头；mask 屏蔽未报数 token。
- `inject_every: int`（1 = 逐 block；k = 每 k 个 block 注入一次，含"仅入口
  层"= inject_every: blocks）。同一实现覆盖整条消融阶梯。
- 与 grad checkpoint 兼容：tokens 经 `_run_block` 透传（与 sht 同模式）。
- 与阶段链兼容：token 分支只存在于 assim 模型；`load_with_new_columns`
  行为不变（token 模块全新随机初始化 = "obs 列随机初始化"的 token 版语义）。

**球谐位置编码**：新增 `sht.py::spherical_harmonics_basis(lat, lon, lmax)`
（任意点实基值），站点基矩阵 (S, 2(l+1)²−1) 注册为 buffer；单测与
MatrixSHT 网格基互验（同位置误差 < 1e-5）。

### 2.3 配置（先 schema 后 yaml）

```yaml
model:
  obs_encoding: grid | token          # A/B 总开关
  obs_token: { d_tok: 128, lmax_pos: 16, heads: 4, inject_every: 1 }
data.synthetic:
  stations_offgrid: true              # 连续站位采样（E2 主开关）
  station_jitter: 0.0                 # 网格站位 + 加性抖动（消融用）
```

### 2.4 论文卖点：attention = 可学习局地化

cross-attention 权重即**流型相关的观测局地化函数**。可视化：某站点
attention 权重的经向剖面 vs LETKF 高斯局地化半径（EnKF/4DVar 基线线已
在 DATA_PLAN §4.5/§6 规划，正好同图对比）。缺失 mask 语义（无 token =
无观测）天然解决现状弱点 2。

### 2.5 实验矩阵（复用 run_osse.py 三对照 + sweep）

| # | 实验 | 设置 | 假设 |
|---|------|------|------|
| E1 | 密度 sweep | OSSE fraction {0.02,0.05,0.08,0.15,0.3} × {grid+mask, token} | 低密度 token 大胜（grid 输入近全 0） |
| E2 | 离网格站点 | stations_offgrid，站位连续随机 | grid 结构性 snap 误差；token 不受影响（新能力） |
| E3 | 报数率 sweep | obs_report_prob {1.0, 0.7, 0.4} | token 天然处理缺失 |
| E4 | L96 A/B | obs_fraction {0.2, 0.5, 0.8} | 分钟级最快迭代，先验证实现 |
| E5 | pilot ERA5 | IGRA/ISD 真实 1900 站真实离网格位置 | 真实大气检验 |

成本：pilot 4608 像素 × 1900 站 × 每 block 一次注意力 ≈ 10⁹ 乘加/层，
4090 无压力。JHTDB（64³=26 万像素）如未来接入需线性注意力或局部窗口
（不在本方案范围）。

---

## 3. B 线：流匹配集合生成

### 3.1 B1 FM head（先行，1–2 周）

**结构**：确定性主干（HuracanSFNO/Huracan1D 原样，deterministic phase
checkpoint 直接复用）+ 小型流匹配头（blocks=2 的小 trunk 或轻 U-Net）。

```
学习速度场 v(x_τ, τ | cond)，线性 FM（rectified flow）：
  x_τ = τ·x1 + (1−τ)·x0,   x0 ~ N(0, I)，τ ~ U(0, 1)
  cond = [确定性预测, x_T]（通道堆叠）
  目标 x1 = target − deterministic_pred（增量分布；residual 语义的推广）
采样：1-step Euler  x1 ≈ x0 + v(x0, 0 | cond)；评估时 n-step 可扫
```

**训练接入**（`phases.py` 按开关切换，阶段名与步数不变）：

- stochastic phase：loss 换 FM loss（head 训练；主干冻结或小 lr 联合）；
- fcst_ar / assim / joint：rollout 用 1-step Euler 采样（前向成本与
  noise-cond 同阶），loss = 采样历史条件下的 FM loss（覆盖条件分布漂移，
  自回归稳定性的关键）+ 可选 CRPS 联合项。

### 3.2 B2 full FM trunk（B1 验证后）

主干整体变为去噪器：输入 = 噪声化未来态（双输出场通道堆叠联合去噪，
12h/call 语义不变）。架构复用点（本方案的独特叙事）：

> `NoiseEmbedder` → 时间步嵌入；池化 g 条件化 `CompressedSpectralFilter`
> 度数权重的机制**原样保留**，含义变为"谱滤波权 h(l) 随去噪阶段变化"
> ——噪声水平相关的球面谱域处理，是论文 A.2 压缩谱滤波机制的自然延伸。

### 3.3 EDA 特有研究问题（论文贡献点）

1. **循环稳定性**：迭代采样 × 120 次 spin-up 调用误差累积 vs noise-cond
   （无人发表过"扩散/流匹配 × 端到端 EDA 循环"的交互行为）；
2. **spread 标定**：ESSR / rank histogram，FM vs noise-cond；
3. **步数–技巧–成本 frontier**：{1,2,4,8,16} 步 sweep（1 步 ≈ 确定性
   速度场一步生成，与 noise-cond 构成理论阶梯）；
4. **A×B 组合**：FM conditioner 消费 obs-token cross-attention 上下文，
   在 EDA 循环内闭环。

### 3.4 配置

```yaml
model:
  ensemble_gen: noise | fm_head | fm      # 三级开关
  diffusion: { head_blocks: 2, head_hidden: 192, sigma_data: 1.0,
               sample_steps_train: 1, sample_steps_eval: 4 }
```

集成点：`modules.py`（NoiseEmbedder 泛化为条件嵌入器）、新
`core/diffusion.py`（FM loss + Euler 采样器）、`rollout.py`
（`member_noise` → `sample_step(model, cond, n_steps)` 抽象）、
`losses.py`（FM loss）、`phases.py`/`module.py`（阶段切换）。

成本：pilot 4 步采样 × 8 成员 × 120 spinup calls ≈ 4× 当前 EDA 推理
耗时，4090 可承受；L96 / OSSE smoke 分钟级。

---

## 4. 文件级改动总表

| 文件 | 改动 | 线 |
|---|---|---|
| `config/schema.py` | ModelCfg: obs_encoding/obs_token/ensemble_gen/diffusion；SyntheticCfg: stations_offgrid/station_jitter；L96 点观测字段 | A/B |
| `core/sht.py` | `spherical_harmonics_basis(lat, lon, lmax)` 工具 | A |
| `core/obs_encoder.py` ⬜新 | ObsTokenEncoder（共享嵌入 + 块内 KV 投影） | A |
| `core/modules.py` / `model.py` / `model1d.py` | block 第三残差分支（inject_every）；`step(..., obs_tokens=...)`；grid 路径 presence 通道 | A |
| `core/diffusion.py` ⬜新 | FM loss + Euler 采样器 | B |
| `core/rollout.py` | obs_tokens 透传；member_noise → sampler 抽象 | A/B |
| `core/losses.py` | FM loss | B |
| `data/turbulence.py` | write_observations 重构（连续站位 + 双渲染）；make_obs_variant 同步 | A |
| `data/lorenz96.py` / `data/obs.py` | 点观测导出（L96 站位=索引；IGRA/ISD 站位=真实经纬度） | A |
| `data/dataset.py` | window() 返回 obs_points（values/mask/站位 buffer） | A |
| `training/phases.py` / `module.py` | ensemble_gen 开关切换 loss/采样 | B |
| `scripts/run_osse.py` | A/B 对照入口 + E1–E3 sweep | A/B |
| `tests/test_units.py` 等 | SH 位置编码互验；mask 语义；cross-attn 形状/经度周期性；FM loss/采样往返；smoke 全链路 | A/B |

## 5. 排期（单人 + 单卡 4090）

| 周 | 内容 |
|---|---|
| 0 | schema 扩展 + obs 缺失 mask 修复（公平 baseline）+ turbulence/L96 站点连续化与点观测契约（双渲染）+ 单测 |
| 1–2 | ObsTokenEncoder（逐 block，inject_every 阶梯）→ L96 A/B（E4）→ OSSE smoke A/B（E1/E2/E3） |
| 2–3 | B1 FM head → L96 → OSSE；ESSR/rank histogram 对比 |
| 4+ | A×B 组合 + sweep 矩阵 + B2 full FM 评估 + pilot ERA5（E5） |

## 6. 风险清单（按优先级）

1. **逐 block 注入显存/速度**：grad checkpoint 已支持；L96 先验证，OSSE
   smoke 实测每 block 开销，必要时退 inject_every=2/入口层（决策 #2 的
   阶梯即对冲）；
2. **参数公平性失控**：token 分支/FM head 参数计入总预算对齐（决策 #8）；
3. **FM head 容量不足**（增量分布表达受限）：B2 full FM 兜底（决策 #3 阶梯）；
4. **严格 schema 拒新配置**：一律先改 schema 再加 yaml（§4 顺序）；
5. **1-step 采样质量不足**：训练可用 2-step（成本 ×2），评估步数 sweep
   给出 frontier，不押注单一设置。
