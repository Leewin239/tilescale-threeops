# 0.1.0 — 2026-11-18

保留 201 步独立复测所用的 AG、RS、AR 实现及其训练路径，整理为 `tilescale_threeops` 包。

| 部分 | 当前保留实现 | 入口 |
| --- | --- | --- |
| AG kernel | 固定 80 CTA、4 个通信 CTA；延后 TMA writeback | `kernels/ag.py`、`runtime.py` |
| AG 路径 | 保存汇集输入及激活，反向复用；打包 FC1/SwiGLU 反向 | `autograd.py` |
| RS kernel | 固定 80 CTA；128×128、K128、3 stages、local-first | `kernels/rs.py`、`runtime.py` |
| RS 路径 | 保存 gate/up；梯度汇集与本地 FC2 权重梯度重叠；将清零提前到 AG 同步之前 | `runtime.py`、`autograd.py` |
| AR kernel | 132 CTA、16 个通信 CTA；8192 归约块，row-major、连续归约，保留完整输出 | `kernels/ar.py`、`runtime.py` |
| AR 路径 | 本地结果视图；融合残差加法与 RMSNorm，供 MLP 消费 | `kernels/residual_norm.py`、`megatron.py` |

原版对照实现随库保留。配置名统一为 `original` 和七种 `paths_*` 配置，默认 `paths_ag_rs_ar`。kernel 内保留了原始函数签名及必要通用参数，库运行时只调用本版记录的参数组合。

旧实验入口、日志、CTA 自动选择探索和物理 SM 分配探索移到工作目录下的 `.archive/2026-11-18-pre-library/`。该目录不属于本发布版本，已加入 `.gitignore`，源码压缩包也不包含它。服务器上的冻结实验记录保持可追溯。

历史性能证据、逐步数据和源文件哈希位于 `results/`。整理后的适配器单独进行运行验证，其结果不替代历史 201 步性能测量。
