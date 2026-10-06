# Huracan 复现（缩小规模可运行版）

复现论文 **"Huracan: A Skillful End-To-End Data-Driven System for Ensemble
Data Assimilation and Weather Prediction"** (Ni et al., arXiv:2508.18486v2)。

官方未发布代码/权重，本项目按论文约束从零实现其全部核心组件，并按本机硬件
（单张 RTX 4090）缩小规模实跑。**复现基准文档为
[docs/HURACAN_REPRODUCTION_GUIDE_REVISED.md](docs/HURACAN_REPRODUCTION_GUIDE_REVISED.md)**：
论文事实 / 高置信推断 / 仓库选择三类来源严格分离（[PAPER]/[INFERRED]/[REPO]），
本仓库的每个关键超参在 `huracan/config/provenance.py` 登记来源。

| 组件 | 论文 | 本复现 (pilot) | 来源 |
|---|---|---|---|
| 网格 | 1°, L=179（网格族未公布） | Gauss-Legendre 48×96 (~3.75°), L=47 | [INFERRED] |
| 时间语义 | 1 次 AR call 前进 12 h，产出 2 个 6 h 场 | 相同 | [PAPER] |
| 变量 | 72 (5×13 层 + 7 地面) | 相同 | [PAPER] |
| 观测通道 | 正文 269 / Table 2 合 334（论文内不一致） | 70（IGRA 探空 + ISD 地面站） | [PAPER]/[REPO] |
| 集合成员 (EDA) | 48 | 8 | [PAPER]/[REPO] |
| spin-up | ≥3 个月（日历驱动，~184 calls） | 60 天 = 120 calls | [PAPER]/[REPO] |
| 训练数据 | 36 年 ERA5 + 15 年观测 | 4 年 ERA5 (2010-2013) | [PAPER]/[REPO] |
| 训练步数 | 347k/149k/14k/46k/13k/2.7k | 40k/15k/3k/5k/2k/1k | [PAPER]/[REPO] |
| SHT 后端 | (SFNO/torch-harmonics) | 自实现精确求积 SHT | [REPO] |

> 论文全文 0 次提及 HEALPix（已核对 v2 原文）；早期版本的 HEALPix 路线已
> 按修订指南删除，可在 git 历史（tag 前的提交）中找到。

## 架构实现（对应论文 Sec. 2.2 / A.2）

`huracan/core/`：

1. **规则球面网格**：Gauss-Legendre 纬度 × 等距经度（`grid.py`）。选择依据：
   nlat = L+1 个 Gauss 节点使 lmax = nlat−1 的 SHT 求积**精确**（equiangular
   需 nlat ≈ 2·lmax），nlon ≥ 2·lmax+1 使经度求和精确，且与 torch-harmonics
   原生布局一致（论文规模 180×360/lmax=179 正是此形式）。
2. **Swin 替换 MLP**（Fig. 2，`swin.py`）：每个 SFNO block 的通道混合 MLP
   换成纬经 2D 窗口 Swin（W-MSA + SW-MSA + MLP，2D 相对位置偏置）。
   经度是周期的 → shift 用真实 wrap（无接缝、无 mask）；纬度不周期 →
   零填充后裁剪。窗口超参论文未公布 [REPO]。
3. **谱滤波压缩**（Sec. A.2，`modules.py`）：`out[l,m,:] = Σ_i h_i(l)·(κ̃_i @ c[l,m,:])`，
   h(l) 由度数的正弦编码拼接全局噪声嵌入过 MLP 得到；k 个共享滤波器
   替代逐度数滤波器（论文 k=15, 12× 参数压缩）。公式无 softmax（论文亦无）。
4. **噪声嵌入 + 条件 LayerNorm**：逐像素高斯噪声 → MLP → 上下文嵌入，
   用于 AdaSpeech 风格的条件 LN（逐通道 scale/shift）。
5. **双时步输入/输出**：输入 (T−6h, T)（含 forcing、观测通道），
   输出 (T+6h, T+12h)。**一次调用前进 12 h、产出两个 6 h 场**（论文
   "6-step=3d / 8-step=4d" 强约束；常量唯一登记在 `rollout.py`）。
   残差式输出 + 输出层零初始化是 [REPO] 稳定训练选择（`model.prediction`
   可切 direct，指南 Sec. 8.6）。
6. **同化模型 = 预报模型 + 观测编码列**（Sec. 2.3）：编码器第一层新
   增观测输入列随机初始化，其余权重继承随机基模型。

## 训练流程（`huracan/training/`，Lightning，论文 Table 3）

```
deterministic  L1, 无噪声              (基模型)
stochastic     + 噪声, CRPS, M=2       (随机基模型)
fcst_ar        6 步自回归微调 = 3 天窗口
assim          同化模型, 8 步自回归 = 4 天窗口 (含观测)
joint          8 步同化 + 8 步预报 联合微调
hybrid         可选: ECMWF 分析微调 (pilot 无 ECMWF 数据, 略过)
```

## 快速开始

```bash
# 0) 环境
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
# GPU 驱动 12.x 用户请用 cu126 版 torch:
# uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# 1) 冒烟测试 (CPU, ~20 秒, 验证全链路)
.venv/bin/python tests/test_smoke.py
.venv/bin/python tests/test_units.py

# 2) 下载数据
#    ERA5: 优先使用 WeatherBench2 公开 zarr (1° 6h, GCS 匿名访问, 免注册, 快):
.venv/bin/python scripts/prepare_era5_wb2.py --config configs/pilot.yaml \
    --years 2010 2011 2012 2013 --workers 6
#    (备选: CDS 官方接口, 需要 ~/.cdsapirc, 服务端 1° 重采样限速严重)
#    .venv/bin/python scripts/download_era5.py --years 2010 2011 2012 2013
#    .venv/bin/python scripts/prepare_era5.py --config configs/pilot.yaml --years 2010 2011 2012 2013
#    观测: IGRA 探空 + ISD 地面站 (免注册直接下载)
.venv/bin/python scripts/download_obs.py --years 2010 2011 2012 2013 \
    --n-igra 400 --n-isd 1500

# 3) 观测预处理到目标网格 (ERA5 状态数据已由 wb2 脚本生成)
.venv/bin/python scripts/prepare_obs.py --config configs/pilot.yaml

# 4) 全部训练阶段 (单卡 4090 约 4-8 小时; 启动时打印通道核算与 provenance)
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/train.py --config configs/pilot.yaml --phase all

# 5) EDA 循环 + 端到端预报 + 加权 CRPS/ESSR/记分卡 (论文 Fig. 3-6)
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_eda.py --config configs/pilot.yaml
```

## 与论文的已知差异（诚实清单）

每个关键超参的来源（PAPER/INFERRED/REPO）登记在
`huracan/config/provenance.py`；训练启动时打印通道核算（含论文 269/334
不一致的显式提示，不静默对齐）。主要差异：

- **卫星观测缺失**：IASI/CrIS/ATMS/SEVIRI 等 L1B 数据需逐个注册申请
  (EUMETSAT/NOAA CLASS)，且论文的超光谱红外/掩星自编码器输入不存在，
  因此 pilot 的观测只有探空 + 地面站（论文 Table 2 中的 in-situ 部分）。
  `huracan/core/autoencoder.py` 已实现并测试，接口就绪，拿到卫星数据即可接入。
- **hybrid 基线**：需要 ECMWF ENS/HRES 初始条件做微调对照，pilot 未包含
  （phase 保留，若获取 ECMWF 数据可直接跑）。
- **SHT 后端**：矩阵 SHT 精确但 O(npix²)，pilot (4608 像素) 可用；
  论文规模 (180×360, lmax=179) 基矩阵约 17 GB，必须注入 torch-harmonics
  （`core/sht.py` 的 `SHTBackend` 协议已对齐，见 `configs/scaleup_inferred.yaml`）。
- **13 层气压层列表**论文未公布（指南 V01），采用覆盖论文评分层的常见
  ERA5 子集 [REPO]。
- **CRPS 估计器 / ESSR 公式 / 空间加权**论文未给（指南 E02-E05）；
  训练用 fair CRPS [REPO]，评估指标支持 Gauss 网格面积加权（默认启用，
  lat-lon 网格不加权会偏向极区）。
- 评估用 ERA5 作真值（论文用各自系统的 analysis）。

## 配置分层（指南 Sec. 19）

| 文件 | 用途 |
|---|---|
| `configs/smoke.yaml` | CPU 冒烟（合成数据，~20 秒全链路） |
| `configs/pilot.yaml` | 4090 单卡实跑（ERA5 2010-2013 + IGRA/ISD） |
| `configs/scaleup_inferred.yaml` | 论文规模（Table 3 全量步数；需 torch-harmonics SHT） |
| `configs/paper_constraints.yaml` | 论文显式约束的参考清单（文档性质，非运行入口） |

## 目录结构

```
huracan/
  core/            框架无关域层（禁止 import lightning/yaml）
    grid.py        Gauss-Legendre 纬经网格 + 面积权重 + 日照/旋转解析工具
    sht.py         精确求积实基 SHT (往返误差 ~5e-6) + SHTBackend 注入协议
    swin.py        纬经 2D 窗口 Swin (经度周期 wrap / 纬度零填充)
    modules.py     压缩谱滤波 / 条件 LN / 噪声嵌入
    model.py       HuracanSFNO (预报/同化两种编码器, SHT 工厂注入, residual/direct)
    rollout.py     时间语义常量 (12h/call) + 双时步展开唯一实现
    losses.py      L1 / fair-CRPS
    metrics.py     CRPS / ESSR / RMSE (可选面积加权)
    io.py          阶段 checkpoint 格式唯一所有者
    autoencoder.py 观测谱压缩自编码器 (IR 16921→32, RO 5994→64)
  config/          dataclass 类型化配置 + provenance 登记 + 通道核算
    schema.py      GridCfg(nlat,nlon,lmax) / EvalCfg(spinup_days,...)
    loader.py      YAML + dot-path overrides
    provenance.py  每个关键超参的 PAPER/INFERRED/REPO 来源表
    report.py      通道核算 (269/334 不一致显式提示)
  data/            数据集(公开 window API)/归一化/合成数据/IGRA/ISD 解析
  training/        Lightning 层 (唯一 import lightning 处)
    module.py      HuracanLitModule (manual optimization, joint 双优化器)
    datamodule.py  LightningDataModule 适配
    callbacks.py   阶段权重链 / 步数预算 / checkpoint
    phases.py      fit_phase: 单入口编排 (Table 3 全部 6 阶段)
  inference/       EDA 循环 + 端到端预报 (日历驱动 spin-up)
  eval/            记分卡绘图
configs/         smoke / pilot / scaleup_inferred / paper_constraints
scripts/         下载、预处理、训练、推理、phase-chain 演示 (瘦 CLI)
tests/           端到端冒烟 + 单元测试 (时间语义/SHT精确性/Swin周期性/provenance)
```

论文 v2 全文提取存于 `data/huracan_v2.txt`（本地参考，不入库）；复现基准
文档为 `docs/HURACAN_REPRODUCTION_GUIDE_REVISED.md`。
