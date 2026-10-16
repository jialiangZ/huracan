# AI4DA 跨域数据方案（DATA PLAN v1.1）

> 2026-10-12 定稿；**2026-10-16 修订：L96（D2）全线移除**（决策撤销，
> 代码/数据/产物/测试清除），跨域谱系由 KF + JHTDB 承担。
> 本方案遵循仓库既有原则：每个数据集 = 一个 nature-run 生成器/下载器 +
> 观测合成 + 与 ERA5 管线逐字节一致的输出布局（state/obs/forcing/norms），
> 下游训练 phases / EDA 循环 / CRPS/ESSR 评估**零改动复用**。

---

## 0. 决策记录（已确认，不再讨论）

| # | 决策项 | 选择 | 主要影响 |
|---|--------|------|----------|
| 1 | ERA5 年份 | ~~2010–2025~~ → **2010–2023（2023 仅至 01-10）** | 2026-10-12 实测：WB2 bucket 最新再网格化 zarr 即 `1959-2023_01_10-*`，时间轴止于 2023-01-10T18:00，无 2024/2025 → 按预案退守（§2.1） |
| 2 | ~~L96 变体~~ | ~~单尺度主线 + 双尺度消融~~ → **2026-10-16 全线移除** | 决策撤销；§3 留移除记录 |
| 3 | KF 实现 | **自研 PyTorch 伪谱求解器** | 可微 → EnKF / 4DVar 基线、ν 参数估计都可行 |
| 4 | KF 分辨率 | **仅 128²** | 训练成本 ~4× 于 64²；放弃分辨率泛化维度（见 §4 回退预案） |
| 5 | KF 粘度 | **多 ν 训练 + 参数估计实验** | 对齐 LEVDA / Xiao et al. 2026 趋势 |
| 6 | JHTDB 数据集 | **仅强迫各向同性湍流 1024³** | 时间演化完整、全域周期、统计均匀 |
| 7 | JHTDB cutout | **64³ 训练 + 128³ 评测** | 训练高效，保留尺寸泛化实验 |
| 8 | JHTDB 变量 | **u, v, w 三分量** | 跨变量约束测试（呼应 z/t/u/v/q 设计） |

---

## 1. 总体架构与前置代码改动

### 1.1 数据源切换（沿用 `data.source` 模式）

```
data.source: synthetic | turbulence | era5 | kolmogorov | jhtdb
```

| 数据集 | 新模块 | 状态 |
|---|---|---|
| ~~L96~~ | ~~`huracan/data/lorenz96.py`~~ | ❌ 2026-10-16 移除 |
| KF | `huracan/data/kolmogorov.py` | ✅ 完成（numpy 伪谱 RK4 + 层流稳态/守恒测试；配置 `data.synthetic.kolmogorov`） |
| JHTDB | `huracan/data/jhtdb.py` | ⬜ 待做（pyJHTDB/REST 下载器 + 本地缓存读取，**无求解器**） |

### 1.2 三处框架改动（**已全部完成**，2026-10-12）

1. ~~**`config/schema.py`**~~ ✅：`JhtdbCfg` 挂在 `data.*`；
   KF 配置为 **`data.synthetic.kolmogorov`（`KolmoCfg`，与 turbulence 并列）**——
   由 kolmogorov.py 的实现锁定，此前计划的 `data.kolmogorov` 重复段已删除；
   `DataCfg` 补 `test_period`。（L96 相关字段已随移除清理。）
2. ~~**`core/rollout.py` 帧语义参数化**~~ ✅：时间语义统一收进
   `huracan/time.py::CallSemantics`（`model.outputs_per_call` 配置，论文
   默认 2 = 一次调用 12h 双场；1 = GenCast 式单场/6h 变体）；
   `EvalCfg.spinup_calls_override / lead_calls_override` 直接以 AR call 计
   （换算入口移到根配置 `cfg.spinup_calls()` / `cfg.lead_calls()`）。
3. **主干网络按几何分派**（同化编码器列、噪声嵌入、条件 LN、phase 编排全部复用）：
   - KF：✅ `core/planar_fft.py`（`grid.type: planar`，PlanarSHT 注入同一 SFNO）
   - JHTDB：3D FNO 或 3D U-Net（64³ 输入，4090 需 grad checkpoint）
   - ERA5 / 球面 OSSE：保留现有 SFNO

### 1.3 布局映射（进入现有 window API 的方式）

| 数据集 | 网格 | upper vars | levels | surface | 通道数 |
|---|---|---|---|---|---|
| KF | 128×128（双周期） | w, u, v | [500] | 无 | 3 |
| JHTDB | 64³ / 128³（三周期） | u, v, w | [500] | 无 | 3 |

---

## 2. D1：ERA5 + IGRA/ISD（2010–2023）

### 2.1 覆盖校验（已完成，2026-10-12）

实测 `gs://weatherbench2/datasets/era5/`：最新再网格化 zarr 为
`1959-2023_01_10-6h-360x181-...zarr`，time 轴 `1959-01-01 → 2023-01-10T18:00`
（93 544 步）。bucket 内**没有任何**覆盖 2024/2025 的 zarr（0.25° full_37
系列同样止于 2023-01-10）。→ **执行退守预案：ERA5 = 2010–2023**，
其中 2023 只有 10 天（40 帧），并入 test 段。将来若要补 2024/2025，
唯一途径是 CDS API（慢），暂不做。

### 2.2 数据规格

| 项 | 规格 |
|---|---|
| 状态 | ERA5 6h，pilot 网格 48×96（保持 3.75°，方法研究分辨率够用） |
| 观测 | IGRA 探空（400 站）+ ISD 地面（1500 站），逐年下载 |
| 划分 | train 2010–2019 / val 2020–2021 / test 2022–2023(01-01–01-10)（按年切，无泄漏） |
| norms | 用 train 段重算（现有 2010–2013 的 norms 作废重生成） |

```bash
.venv/bin/python scripts/prepare_era5_wb2.py --config configs/pilot.yaml \
    --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 \
            2020 2021 2022 2023 --workers 6
.venv/bin/python scripts/download_obs.py --years 2010..2023 \
    --n-igra 400 --n-isd 1500
.venv/bin/python scripts/prepare_obs.py --config configs/pilot.yaml
```

### 2.3 预算

- 状态数据 ≈ 2 GB/年（48×96 网格 float32）→ 14 年 ≈ 30 GB；
  原始 chunk 缓存（raw_par_*）另计，翻倍预留磁盘 ~130 GB。
- 观测下载 1–3 天（挂机，可与 D2/D3 开发并行）。

---

## 3. D2：Lorenz-96 —— ❌ 已移除（2026-10-16）

> 决策撤销：`huracan/data/lorenz96.py`、`core/model1d.py`（mlp1d 主干）、
> `configs/l96*.yaml`、`scripts/generate_lorenz96.py`、`scripts/run_l96_controls.py`、
> `tests/test_lorenz96.py`、`data/lorenz96*` 及全部 runs 产物均已删除。
> 原规格（K=40/F=8 主线 + K=36/J=10 双尺度消融、偶数变量观测 σ=1、
> 三对照协议）见 git 历史（commit 5c3a4eb）。跨域低维混沌验证改由
> KF 64² OSSE（分钟级）承担。

---

## 4. D3：Kolmogorov flow（自研伪谱，128²，多 ν）

> **状态（2026-10-12）**：求解器与管线已由 `huracan/data/kolmogorov.py` 实现
> （numpy 伪谱 + 2/3 去混叠 + RK4，强迫幅值 warm-up 校准；配置在
> `data.synthetic.kolmogorov`（KolmoCfg：nu/kf/u_rms/dt/dt_frame/warmup_frames），
> 平面几何走 `grid.type: planar` + PlanarSHT）。快速 OSSE 版
> `configs/osse_kolmo.yaml`（64²，ν=2e-3，kf=4）已跑通；**本节剩余工作 =
> 扩展 KolmoCfg 到 128² 多 ν 矩阵 + ν 条件通道/参数估计**（决策 #4/#5），
> 帧语义沿用其 dt_frame=0.25（1 AR call = 2 帧 = 0.5 t.u.）。

### 4.1 方程与数值设置

```
∂ω/∂t + J(ψ, ω) = ν∇²ω − r·ω + f,   ω = ∇²ψ
f = A·sin(k_f·y)·ê_x          （Kolmogorov 强迫）
```

| 项 | 规格 | 备注 |
|---|---|---|
| 域 | [0, 2π)² 双周期，128×128 | kmax ≈ 42（2/3 去混叠） |
| k_f | **实现时对标 ADDA / Exponax 的 KF 默认后锁死**（常见 k_f=8），A 校准到能量稳态 | 保证与 ADDA 可比 |
| drag r 或超粘性 | 对标 ADDA 二选一（无耗散机制则能量堆积到最大尺度，非稳态） | 同上 |
| 时间积分 | RK4；内部子步以 CFL 校准（预期 dt≈0.005–0.01）；**1 帧 = 0.1 t.u.** | 可微（torch 全图内） |

### 4.2 ν 矩阵（决策 #5：多 ν + 参数估计）

| 用途 | ν 值 | 轨迹 |
|---|---|---|
| 训练 | 1e-5, 3e-5, 1e-4, 3e-4, 1e-3（对数均匀 5 档） | 每档 2 条 × 2000 帧 |
| 验证 | 同训练 5 档 | 每档 200 帧 |
| 测试-插值 | ν ∈ 两相邻训练档中点（取 2 个） | 各 300 帧 |
| 测试-外推 | 3e-6（若谱分辨率允许，见验证）与 3e-3 | 各 300 帧 |

- warmup：每条轨迹先丢 500 帧（统计平衡）。
- **ν 条件化设计**：ν 作为额外标量 forcing 通道进入模型（占一个通道位），
  单一模型跨 ν 训练；参数估计实验时把 ν 当 EDA 增广状态（或对 ν 通道做
  扰动-校正），直接对标 LEVDA 的"unknown viscosity"设定。

### 4.3 状态与观测

- 状态通道：ω（涡度）、u、v（由 ψ 解析求出，无辐散），共 3 通道；
- 观测（默认）：固定站点网，fraction=0.08（对齐球面 OSSE 的 369 站设置），
  噪声 = 通道 std 的 15%，每帧报数；密度 sweep {0.02, 0.05, 0.08, 0.15, 0.3}、
  噪声 sweep {0.05, 0.15, 0.3}（复用 `make_obs_variant` 机制）。

### 4.4 验证测试（`tests/test_kolmogorov.py`）

1. ν=r=0 短时间能量守恒 < 1e-8（对齐 turbulence.py 的标准）；
2. warmup 后 E(t) 漂移 < 5%（稳态判定）；
3. **谱尾不堆积在 kmax**（ν=1e-5 档若堆尾 → ν 下限上调至 3e-5，§4.2 外推档同步调整）；
4. 识别谱斜率段（k_f 峰 + 大尺度 k^{-3}/k^{-5/3}），与文献 KF 谱对照；
5. 可微性单测：autograd 对 ν 的梯度有限。

### 4.5 基线与预算

- 基线：EnKF / LETKF（自研求解器即正演模式，几乎零成本）+ **4DVar**
  （可微求解器直接反传，ADDA 同款对比点）+ 气候态/persistence；
- 存储：22.5k 帧 × 197 KB ≈ **4.5 GB**（float32）；
- 128² 训练慢的**回退预案**（决策 #4 的对冲）：64² 预训练 → 128² 微调，
  或动态降 batch；此回退不影响已锁定决策，只是工程手段。

---

## 5. D4：JHTDB 强迫各向同性湍流（Isotropic1024fine）

### 5.1 事实（以数据库元数据为准）

1024³ DNS，[0, 2π)³ 三周期，完整时间演化 ≈ 5 个大涡翻转时间，库存
Δt≈0.002（数据库单位）。**快照型数据集（4096³/8192³/32768³/Channel5200）
一律不用。**

### 5.2 获取流程（顺序执行）

1. turbulence.pha.jhu.edu 注册免费 **auth token**；
2. 浏览器逐点查询 / SciServer 熟悉场（探索阶段零下载）；
3. `scripts/download_jhtdb.py`：优先新 **REST + Zarr 64³ 桶**接口（老数据集
   是 SOAP + Z-curve），cutout 批量下载过夜，缓存到 `data/jhtdb/`；
4. 下载后立即转 float16 存储（训练时转回 float32），减半磁盘。

### 5.3 cutout 矩阵（决策 #7、#8）

| 集合 | 规格（空间 × 时间） | 数量 | 原点规则 |
|---|---|---|---|
| Train | 64³ × 128 帧（stride 5 库存步 → Δt=0.01，跨度 1.28 t.u.） | 64 | 随机原点，任意两原点 Chebyshev 距离 ≥128（无重叠） |
| Val | 同上 | 8 | 与 train 不相交位置 |
| Test-A 空间 | 同上 | 8 | 训练未见位置 |
| Test-B 时间 | 同上 | 4 | 晚段独立时间窗 |
| Test-C 尺寸 | 128³ × 64 帧 | 4 | 空间+时间均 held-out |

- 存储：float32 ≈ 40 GB → **float16 ≈ 20 GB**；
- 时间起点：总库存步数以元数据核实（预期 ~5000 步量级），各 cutout 起点
  均匀散布在可用范围内（时间上允许重叠——空间不相交才是独立样本维度）。

### 5.4 观测算子（合成，复用 OSSE 机制）

| 变体 | 设计 | 用途 |
|---|---|---|
| 默认：点传感器 | 随机体素 1%（可 sweep 0.2–5%），σ=10% 通道 std，每帧报数 | 主实验（热线/探针类比） |
| 粗网格 | stride-4 降采样 + 噪声 | 卫星/成像类比 |
| PIV 平面 | 1–2 个随机平面全帧 | 激光切片类比 |

### 5.5 验证与评估

- 物理验证：三分量 rms 各向同性（差 <5%）；E(k) 惯性段斜率 ≈ −5/3，
  与 Graham et al. (2016) 发表统计对照；时间自相关长度 → 据此定 DA 窗口；
- 评估：RMSE/CRPS/ESSR（复用）+ **谱误差指标**（分析场 vs 真值 E(k)）；
- 无求解器 → 无 EnKF/4DVar 基线，这正是叙事卖点：**端到端数据驱动同化
  不需要可复跑正演模式**；
- 引用义务：Graham et al. (2016) + 数据集页（写入论文与仓库 README）。

---

## 6. 统一实验协议（各数据集一致）

1. 三对照：**with-obs / zero-obs / fcst-spinup**（复用 `run_osse.py`，同训练
   同种子）——"有观测时分析误差随循环下降"是每个数据集的第一判据；
2. 指标：RMSE / fair-CRPS / ESSR + 每数据集物理指标
   （KF：E(k) + ν̂ 误差；JHTDB：谱误差；ERA5：记分卡）；
3. 每个设置 3 seeds；观测密度 × 噪声 sweep（每数据集复用同一 nature run）；
4. 基线总表：气候态、persistence（全数据集）+ EnKF/LETKF（KF）+ 4DVar（KF）。

---

## 7. 交付物与排期

### 7.1 新增文件清单

```
huracan/data/kolmogorov.py   (生成逻辑内置, 待 CLI)          configs/osse_kolmo.yaml ✅
huracan/data/jhtdb.py        scripts/download_jhtdb.py       configs/jhtdb.yaml ⬜
huracan/core/planar_fft.py (平面谱变换) ✅
tests/test_kolmogorov.py ✅  tests/test_jhtdb.py ⬜
```

（顺序注意：schema/loader 先行 → 数据模块 → 配置，严格校验会拒绝超前配置。）

### 7.2 排期（单人 + 单卡 4090）

| 周 | 内容 |
|---|---|
| 0（并行挂机） | ERA5 + 观测下载；WB2 年份覆盖校验 |
| 1–2 | 框架改动（§1.2）（L96 原排期已随移除作废） |
| 3–5 | KF 求解器 → 验证测试 → 数据生成 → 训练 + 4DVar 基线 |
| 5–8 | JHTDB token/探索 → cutout 下载 → 3D 主干 → 实验 |
| 8+ | 跨数据集横向图表 + 论文实验章 |

---

## 8. 风险清单（按优先级）

1. ~~**WB2 zarr 未覆盖 2024/2025**~~ → **已实测确认并退守 2010–2023**（2026-10-12，§2.1）；
2. **JHTDB 库存步数/带宽不及预期** → 元数据先行核实，下载过夜分批；
3. **KF ν=1e-5 谱尾撞 kmax** → ν 下限上调 3e-5（验证测试 3 兜底）；
4. **128² KF 训练过慢** → 64² 预训练 + 128² 微调（工程回退，不改决策）；
5. **3D 主干显存不足** → 64³ 小 batch + grad checkpoint（仓库已支持）；
6. **严格 schema 拒新配置** → 一律先改 schema 再加 yaml（§7.1 顺序）。
