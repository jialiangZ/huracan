# Huracan 复现（缩小规模可运行版）

复现论文 **"Huracan: A Skillful End-To-End Data-Driven System for Ensemble
Data Assimilation and Weather Prediction"** (Ni et al., arXiv:2508.18486)。

官方未发布代码/权重，本项目按论文描述从零实现其全部核心组件，并按本机硬件
（单张 RTX 4090）缩小规模实跑：

| 组件 | 论文 | 本复现 (pilot) |
|---|---|---|
| 网格 | HEALPix nside=64 (~1°), L=179 | nside=16 (~3.7°), L=32 |
| 变量 | 72 (5×13 层 + 7 地面) | 相同 |
| 观测通道 | 269（含卫星 L1B） | 70（IGRA 探空 + ISD 地面站） |
| 集合成员 (EDA) | 48 | 8 |
| 训练数据 | 36 年 ERA5 + 15 年观测 | 4 年 ERA5 (2010-2013) |
| 训练步数 | 347k/149k/14k/46k/13k | 40k/15k/3k/5k/2k |
| SHT 后端 | (SFNO/torch-harmonics) | 自实现实基最小二乘 SHT |

## 架构实现（对应论文 Sec. 2.2 / A.2）

`huracan/modules.py` + `huracan/model.py`：

1. **Swin 替换 MLP**（Fig. 2）：每个 SFNO block 的通道混合 MLP 换成
   face-wise Swin Transformer（HEALPix 12 个面各为 n×n 图像，W-MSA +
   SW-MSA + MLP，相对位置偏置）。简化点：shift 的 wrap-around 发生在
   面内而非真实 HEALPix 邻接。
2. **谱滤波压缩**（Sec. A.2）：`out[l,m,:] = Σ_i h_i(l)·(κ̃_i @ c[l,m,:])`，
   h(l) 由度数的正弦编码拼接全局噪声嵌入过 MLP 得到；k 个共享滤波器
   替代逐度数滤波器（论文 k=15, 12× 参数压缩）。
3. **噪声嵌入 + 条件 LayerNorm**：逐像素高斯噪声 → MLP → 上下文嵌入，
   用于 AdaSpeech 风格的条件 LN（逐通道 scale/shift）。
4. **双时步输入/输出**：输入 (T−6h, T)（含 forcing、观测通道），
   输出 (T+6h, T+12h) 的残差增量。
5. **同化模型 = 预报模型 + 观测编码列**（Sec. 2.3）：编码器第一层新
   增观测输入列随机初始化，其余权重继承随机基模型。

## 训练流程（`huracan/train/trainer.py`，论文 Table 3）

```
deterministic  L1, 无噪声              (基模型)
stochastic     + 噪声, CRPS, M=2       (随机基模型)
fcst_ar        6 步自回归微调
assim          同化模型, 8 步自回归 (含观测)
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

# 1) 冒烟测试 (CPU, ~1 分钟, 验证全链路)
.venv/bin/python tests/test_smoke.py

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

# 3) 观测预处理到 HEALPix (ERA5 状态数据已由 wb2 脚本生成)
.venv/bin/python scripts/prepare_obs.py --config configs/pilot.yaml

# 4) 全部训练阶段 (单卡 4090 约 4-6 小时)
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/train.py --config configs/pilot.yaml --phase all

# 5) EDA 循环 + 端到端预报 + CRPS/ESSR/记分卡 (论文 Fig. 3-6)
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_eda.py --config configs/pilot.yaml
```

## 与论文的已知差异（诚实清单）

- **卫星观测缺失**：IASI/CrIS/ATMS/SEVIRI 等 L1B 数据需逐个注册申请
  (EUMETSAT/NOAA CLASS)，且论文的超光谱红外/掩星自编码器输入不存在，
  因此 pilot 的观测只有探空 + 地面站（论文 Table 2 中的 in-situ 部分）。
  `huracan/autoencoder.py` 已实现并测试，接口就绪，拿到卫星数据即可接入。
- **hybrid 基线**：需要 ECMWF ENS/HRES 初始条件做微调对照，pilot 未包含
  （phase 保留，若获取 ECMWF 数据可直接跑）。
- **SHT 后端**：实基最小二乘 SHT 精确但 O(npix²)，nside=64 时需换
  torch-harmonics（接口已对齐 `forward/inverse`）。
- **Swin 面内 wrap-around**、**度数权重无 softmax**（论文公式亦无）等
  小简化在代码注释中标出。
- 评估用 ERA5 作真值（论文用各自系统的 analysis）。

## 目录结构

```
huracan/
  grid.py        HEALPix NESTED 网格、日照 forcing、旋转
  sht.py         实基球谐变换 (往返误差 ~1e-6)
  modules.py     压缩谱滤波 / Swin 块 / 条件 LN / 噪声嵌入
  model.py       HuracanSFNO (预报/同化两种编码器)
  autoencoder.py 观测谱压缩自编码器 (IR 16921→32, RO 5994→64)
  data/          数据集、归一化、合成数据、IGRA/ISD 解析
  train/         阶段训练器 + L1/CRPS 损失
  inference.py   EDA 循环 + 端到端预报
  eval/          CRPS/ESSR/RMSE + 记分卡绘图
configs/         smoke (CPU 冒烟) / pilot (4090 实跑) / paper (论文规模)
scripts/         下载、预处理、训练、推理入口
tests/           端到端冒烟测试
```
