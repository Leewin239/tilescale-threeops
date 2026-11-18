# TileScale ThreeOps

AG+GEMM、GEMM+ReduceScatter、GEMM+AllReduce 的固定资源优化，以及 Qwen3 Megatron 训练路径。

当前发布版本为 0.1.0，来源于 2026-11-18 的 201 步独立复测实现。整理后的库提供八种配置，分别启用 AG、RS、AR、两两组合及三路联合。默认启用三路联合。

保留实现与版本清理范围见 [CHANGELOG.md](CHANGELOG.md)，复测记录见 [results](results/README.md)。

本版保留固定资源下的 kernel 与训练路径优化。配置由用户显式选择；本版不包含自动生成候选、自动调优或物理 SM 划分实现。

## 环境

已验证硬件为单机 H100 80GB，TP2、NVSwitch multicast、BF16。模型为 Qwen3-1.7B，28 层，4096 tokens，SP，micro/global batch 均为 1。需要支持分布式 allocator、TMA 和 multimem 的 TileScale/TileLang 安装，以及 Pai-Megatron-Patch 和 Transformer Engine。

TileScale 上游提交为 `71e71457d40ed6e6a2dad297936bfdbff21d104b`，Pai-Megatron-Patch 为 `a098ca5acbdeaf7172ce0393fa309b39a77506db`。详细复测信息见 results。

## 安装和运行

```bash
python -m pip install -e . --no-deps
python scripts/build_native.py

export PAI_ROOT=/path/to/Pai-Megatron-Patch
export QWEN_HF=/path/to/Qwen3-1.7B
export QWEN_CHECKPOINT=/path/to/megatron-checkpoint
export PYTHON=python
export THREEOPS_CONFIG=paths_ag_rs_ar

bash scripts/train.sh 0,1 4096 /path/to/open-web-math_text_document
```

检查点使用 Pai 的 Qwen3 转换器生成，采用 torch_dist 格式。数据路径指向 Megatron MMAP 数据前缀。模型权重和数据由使用者提供。环境中已安装 TileScale 时无需设置额外 Python 路径。若沿用独立安装，可通过 `TILESCALE_PYTHON_PATH` 指定其 site-packages 路径。

配置包括 `original`、`paths_ag`、`paths_rs`、`paths_ar`、`paths_ag_rs`、`paths_ag_ar`、`paths_rs_ar`、`paths_ag_rs_ar`。AG 使用 80 CTA 和 4 个通信 CTA，RS 使用 80 CTA，AR 使用 H100 的 132 CTA 和 16 个通信 CTA。

每层首次调用默认与 Megatron 原生输出和梯度对照。完整训练步包括优化器更新。默认运行 201 步，统计时排除前 20 步，超时由 `MEGATRON_TIMEOUT_SECONDS` 控制。

启动器在超时时清理训练子进程，包括 torchrun 创建的独立会话。首次编译默认使用一个 TorchInductor 编译线程，可通过 `TORCHINDUCTOR_COMPILE_THREADS` 调整。

独立 kernel 延迟测试使用安装了 TileScale 的 Python 环境运行。

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 python -m tilescale_threeops.benchmark ag
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 python -m tilescale_threeops.benchmark rs
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 python -m tilescale_threeops.benchmark ar
```

完整配置复测沿用上面的环境变量，并分别执行两个镜像区组。

```bash
bash scripts/compare.sh path-singles 0,1 /path/to/open-web-math_text_document
bash scripts/compare.sh path-doubles 0,1 /path/to/open-web-math_text_document
python -m tilescale_threeops.analyze path-singles runs/path-singles.RUN_ID --steps 201 --warmup 20
```

复测脚本每两秒检查同卡进程。测量窗口出现外部进程时拒绝该区组计时。日志保存在 runs 下。

## 当前性能证据

冻结实现的纯 kernel 延迟下降约为 AG 2.2%、RS 9.3%、AR 7.9%。201 步独立训练复测的三路联合对原版配对速度比为 1.1354–1.1781。联合对双路的合并比较均为正，但对 AG+RS 的第一次重复接近持平，第二次更快。数值和性能结论适用于上述已测配置。

代码整理后的运行验证记录及历史结果位于 results。

运行 `python scripts/check_results.py` 可从保存的逐步数据重新核对历史配对速度比；该命令不启动训练。

测试环境为 Python 3.10.20、PyTorch 2.4.0+cu121、TileLang 0.0.2 的 TileScale 分布式构建、Transformer Engine 1.12.0+7a7225c4。设备 kernel 使用 CUDA 12.8 工具链编译。依赖源码、安装方式与本机 CUDA 工具链需配套。

## 来源

kernel 基于 TileScale 的分布式示例修改。原始版权及许可保留在 third_party/LICENSE.TileScale。本项目新增代码的发布许可由仓库所有者决定，详见 third_party/NOTICE.md。
