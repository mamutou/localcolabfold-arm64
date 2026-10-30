# LocalColabFold — ARM64 (aarch64) 版

在**自己的 64 位 ARM（aarch64）Linux 机器**上跑 [ColabFold](https://github.com/sokrypton/ColabFold)：没有 Colab 的 90 分钟/12 小时超时，也没有 GPU 配额。

本仓库是 [YoshitakaMo/localcolabfold](https://github.com/YoshitakaMo/localcolabfold) 的 fork。上游的安装脚本是为 x86_64 Linux 和 macOS 写的：下载 `Miniforge3-Linux-x86_64.sh`、装 x86_64 命名规则的 `jax[cuda12]` 轮子、附带 Intel/M1 Mac 专用脚本。在 ARM64 服务器（NVIDIA DGX Spark / GB10、Grace Hopper、Graviton、Ampere Altra + GPU……）上，这套流程在 conda 引导那一步就直接断掉，`colabfold_batch` 根本装不出来。

这个 fork 保留上游的目录结构和命令行用法，只把与架构绑定的部分换成 ARM64 原生版本，并把那些"不踩到就不知道"的 aarch64 坑写进文档。

- 上游项目：<https://github.com/YoshitakaMo/localcolabfold>
- 本 fork：<https://github.com/mamutou/localcolabfold-arm64>
- English: [README.md](README.md)

## 实测环境

本仓库的安装脚本是在这台机器上开发和验证的（单台 NVIDIA DGX Spark）：

| 项目 | 值 |
| --- | --- |
| 机器 | NVIDIA DGX Spark，GB10（Blackwell，compute capability 12.1），128 GB 统一内存 |
| 系统 | Ubuntu 24.04.4 LTS，`aarch64` |
| 驱动 / CUDA | 580.173.02 / CUDA 13.0.3（`/usr/local/cuda-13.0`） |
| Python | 3.10.19（conda-forge，Miniforge3-Linux-aarch64） |
| ColabFold | 1.5.5 |
| JAX | `jax`/`jaxlib` 0.6.2（另装了 `jax_cuda12_plugin` 0.6.2，在 sm_121 上**不可用**，原因见下） |
| 其他版本 | `numpy` 2.2.6、`openmm` 8.2.0、`tensorflow` 2.20.0 |

单体与多体（`alphafold2_ptm` / `alphafold2_multimer_v3`，默认 MSA 服务器）的预测都在这台机器上真实跑完过，所以 **CPU 路径是端到端验证过的**；**GPU 路径没有**，见 [GPU 加速](#gpu-加速实验性)。

## 相比上游改了什么

| 文件 | 改动 |
| --- | --- |
| `install_colabbatch_linux.sh` | 自动识别 `x86_64` / `aarch64` 并选对应的 Miniforge3；aarch64 分支安装 ARM64 原生工具链、锁定一组已验证的 JAX 版本、打上 ARM64 补丁，并生成 `setup_env.sh`。 |
| `setup_env.sh` | 新增。运行 `colabfold_batch` 前 `source` 它，设置 `CUDA_HOME`、`LD_LIBRARY_PATH`（含 `/usr/lib/aarch64-linux-gnu`）以及一个在 CUDA 不可用时不会把程序搞挂的 `JAX_PLATFORMS`。 |
| 更新脚本下载地址 | 上游安装脚本从 `YoshitakaMo/localcolabfold` 拉 `update_*.sh`，现在改为从本 fork 拉，fork 装完不会又退回上游脚本。 |
| `README.md` / `README_zh.md` | 针对 ARM64 重写（英文 / 本文件）。 |
| `README_ja.md` | 删除——那是上游 README 的旧日文副本，早已和脚本对不上。 |
| `beta_install_colabbatch_linux.sh`、`beta_update_linux.sh` | 删除——1.2 之前的遗留脚本，锁定 python 3.8/3.9 和 **x86_64** 的 CUDA-11 `jaxlib` 轮子，在 aarch64 上没有任何一步能跑，在 x86_64 上也早已失效。 |
| macOS 安装脚本 | 原样保留，纯 Apple 平台，与 ARM64 Linux 无关；Mac 用户请用上游。 |

除上述文件外（`colabfold_batch` 的行为、输入格式、命令行参数、模型权重）全部是上游 ColabFold 原样，未做修改。

## 安装（Linux ARM64）

### 前置条件

1. `curl`、`git`、`wget`：

   ```bash
   sudo apt -y install curl git wget
   ```

2. GNU 编译器 **12.0 以上**（openmm 8.2 的 `--amber` relax 需要 `GLIBCXX_3.4.30`）：

   ```bash
   gcc --version
   ```

   Ubuntu 24.04 自带的 `gcc-13` 就够。

3. 如需使用 GPU，安装与驱动匹配的 CUDA **13.x**。用 `nvcc --version` 查，**不要**用 `nvidia-smi`：

   ```bash
   $ nvcc --version
   Cuda compilation tools, release 13.0, V13.0.88
   ```

4. 磁盘：conda 环境约 3 GB，AlphaFold2/ColabFold 权重约 4 GB，结果另算。

### 安装

```bash
mkdir -p ~/work && cd ~/work          # 装完不要再移动这个目录
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/install_colabbatch_linux.sh
bash install_colabbatch_linux.sh
```

5～15 分钟后得到 `~/work/localcolabfold/`：

```
localcolabfold/
├── conda/                 # Miniforge3 (aarch64)
├── colabfold-conda/       # python 3.10 环境：colabfold、jax、openmm、kalign2、hhsuite、mmseqs2
├── colabfold/             # 模型权重
├── setup_env.sh           # 预测前 source 它
└── update_linux.sh
```

安装期间别断网，并**看一眼日志**。失败的话最简单的做法是删掉 `localcolabfold` 目录重装——脚本不支持断点续装。

### 配置 shell

```bash
source ~/work/localcolabfold/setup_env.sh
```

（`setup_env.sh` 可以重复 source；想永久生效就加到 `~/.bashrc`。它会导出 `PATH`、`CUDA_HOME`/`LD_LIBRARY_PATH` 和一个安全的 `JAX_PLATFORMS`。）

### 第一次预测

```bash
cat > test.fasta <<'EOF'
>sp|P61823
MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN
LTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPN
CAYKTTQANKHIIVACEGNPYVPVHFDASV
EOF

colabfold_batch test.fasta out/
```

结果（`*_unrelaxed_rank_001_*.pdb`、`*_scores_rank_001_*.json`、`*_plddt.png`、`*_pae.png`）写在 `out/`。要模板和 AMBER 能量最小化，加 `--templates --amber`。

## 安装后自检

```bash
source ~/work/localcolabfold/setup_env.sh
python -c "import importlib.metadata as m; print(m.version('colabfold'), m.version('jax'))"
python -c "import jax; print(jax.devices())"
colabfold_batch --help | head -5
```

在这套版本下应输出 `1.5.5 0.6.2` 和 `[CpuDevice(id=0)]`（为什么不是 GPU 见下一节）。

## GPU 加速（实验性）

**一句话：用上面锁定的版本，ColabFold 在 GB10 上走 CPU。** 在那个环境里要 CUDA 会直接报：

```
RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA
```

原因：`jaxlib` 0.6.2 / `jax_cuda12_plugin` 0.6.2 里的 cubin 只覆盖 Blackwell 之前的 SM，而 GB10 是 `sm_121`。插件能加载，但找不到匹配设备，于是 JAX 退回 CPU（或直接报错）。改 `LD_LIBRARY_PATH` 没有任何用——限制在轮子内部。

ColabFold 之外的情况已经变了：JAX 从 **0.7.2** 起发布带 `linux_aarch64` 轮子的 `jax[cuda13]`（CUDA 13 plugin + PJRT），但对 Python 的要求是 **cp311 起**（没有 cp310 轮子）。在这台机器上验证过下面这条能枚举出 GPU：

```bash
# 单独的环境——不要动 ColabFold 的 env
python3.12 -m venv ~/cf_probe/venv          # Ubuntu 24.04 自带 python3.12
~/cf_probe/venv/bin/pip install "jax[cuda13]==0.7.2"
~/cf_probe/venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
# 0.7.2 [CudaDevice(id=0)]
```

`colabfold[alphafold-minus-jax]`（1.6.3）在那个 venv 里也能正常 import。**未验证**的是：在 jax 0.7.x 下跑完一次端到端预测——ColabFold 1.5/1.6 是照 jax 0.4/0.5 的 API 写的，模型代码里仍可能踩坑。如果你需要 GPU 吞吐，`jax[cuda13]` 是值得尝试的路，但请先拿一条短序列验证，再放心跑批量。

打通之后想用 GPU：导出 `JAX_PLATFORMS=cuda`（或不设、让 JAX 自动探测），跑长任务前先确认 `jax.devices()` 是 `CudaDevice`。

## DGX Spark 上的实测容量

合成序列，`--msa-mode single_sequence` 与自建 A3M（深度 512）两种配置，1 recycle、1 个模型、1 seed、bfloat16、128 GB 统一内存、不做 relax：

| 序列长度 | single-sequence | MSA 深度 512 |
| --- | --- | --- |
| 1000 | — | ✅ |
| 1500 | ✅ | ✅ |
| 2000 | — | ✅ |
| 2500 | ✅ | ✅ |
| 2800 | ✅ | ✅ |
| 3000 | ✅ | ✅ |
| 3200 | ✅ | ✅ |
| 3400 | ✅（约 48 分钟） | ✅ |
| 3450 | ❌ OOM | ❌ OOM |

也就是说这台机器单次预测按 **L ≲ 3400** 规划；上限还会随 MSA 深度、`--num-recycle`、模型个数变化。

## aarch64 已知问题（本 fork 的处理方式）

1. **`jax[cuda12]` 看不到 Blackwell GPU。** 见 [GPU 加速](#gpu-加速实验性)。上游 x86_64 安装脚本里那句 `jax[cuda12]==0.5.3` 在 GB10 上**不是**可用的 GPU 配方。
2. **`ModuleNotFoundError: No module named 'tpu_info'`**：某些安装组合下 import `colabfold.batch` 会失败。安装脚本把该 import 包进 `try/except ImportError`。
3. **`jax.config.update('jax_enable_x64', False)` 被插到 `from __future__ import ...` 之前**会让 `batch.py` 变成 `SyntaxError`。补丁把 float32 配置插在所有 `__future__` 导入之后，并对 `batch.py` 做一次字节码编译以确认。
4. **残留的 `JAX_PLATFORMS=cuda`**：一旦 CUDA 插件不可用，之后每次运行都会硬失败。`setup_env.sh` 只在确实存在可用 CUDA JAX 时才设 `cuda`，否则清空它。
5. **aarch64 上 cuDNN/CUDA 库在 `/usr/lib/aarch64-linux-gnu`**，不是 `/usr/lib/x86_64-linux-gnu`。`setup_env.sh` 两个路径都加。
6. **`linux-aarch64` 的包可用性和 x86_64 不一样**。2026-10 在 conda-forge/bioconda 与 PyPI 上核对过：

   | 包 | `linux-aarch64` |
   | --- | --- |
   | `openmm`（conda-forge） | ✅ 有 8.2.x |
   | `hhsuite` = 3.3.0（bioconda） | ✅ |
   | `mmseqs2`（bioconda） | ✅ |
   | `kalign2` 2.04 / `kalign` 3.x | ✅ |
   | `jaxlib` / `jax-cuda12-*` | ✅（cp310+） |
   | `jax-cuda13-*` | ✅ 从 0.7.2 起，但**只到 cp311+** |
   | `colabfold`（PyPI sdist） | ✅ 可从 git/pip 安装 |

## 输入格式

`colabfold_batch` 接受 FASTA、A3M、CSV/TSV，或一个装着 FASTA/A3M 的目录。

```fasta
>sp|P61823
MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN
LTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPN
CAYKTTQANKHIIVACEGNPYVPVHFDASV
```

多体用 `:` 分隔各链：

```fasta
>3KUD_RasRaf_complex
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQ
YMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIP
YIETSAKTRQGVEDAFYTLVREIRQH:
PSKTSNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAAS
LIGEELQVDFL
```

CSV：

```csv
id,sequence
5AWL_1,YYDPETGTWY
```

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--amber` | AMBER relax；`--num-relax N` 限定只 relax 前 N 个模型 |
| `--use-gpu-relax` | AMBER 跑在 GPU（仅 NVIDIA + CUDA） |
| `--templates` | 使用 PDB 模板；`--custom-template-path DIR` 限定只用本地模板 |
| `--num-recycle N` | recycle 次数，越多越好越慢，默认 3 |
| `--num-seeds N` / `--random-seed N` | 多个 seed 采样，CPU 机器上用时间换多样性 |
| `--max-msa 512:1024` | 限制 MSA 深度——**这台机器上最省内存的一招** |
| `--use-dropout` | 从模型不确定性中采样 |
| `--overwrite-existing-results` | 覆盖已有结果 |

完整列表：`colabfold_batch --help`。

## 如何更新

ColabFold 本体迭代很快，更新环境（不是安装目录）：

```bash
cd ~/work/localcolabfold
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/update_linux.sh -O update_linux.sh
chmod +x update_linux.sh
./update_linux.sh .
```

它会重装 ColabFold 并重新锁定本 fork 验证过的 JAX/numpy 版本。如果你为了 GPU 手动升过 JAX，就别跑这个更新脚本，或者先改掉里面的版本锁。

更新本 fork 的脚本：

```bash
cd ~/work
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/install_colabbatch_linux.sh -O /tmp/new_installer.sh
diff install_colabbatch_linux.sh /tmp/new_installer.sh
```

## FAQ

- **需要下载 PDB70、BFD、Uniclust30、MGnify 这些大库吗？** 不需要。MSA 由 MMseqs2 Web 服务器生成，和 ColabFold 一样。要离线跑，就用 ColabFold 仓库的 `setup_databases.sh` 建库，再用 `colabfold_search`。
- **能用多张 GPU 吗？** 不能——AlphaFold/ColabFold 单设备推理。多卡机器上用 `CUDA_VISIBLE_DEVICES` 给每个进程分一张。
- **支持 Apple Silicon（M1/M2/M3）吗？** 本 fork 不涉及，那是上游的 `install_colabbatch_M1mac.sh`（本仓库原样保留）。注意它默认那套 `jax==0.4.23` 已经很旧。
- **ARM64 Windows 的 WSL2 能跑吗？** 未测试。上游关于 WSL2 大小写敏感目录的说明仍然适用。
- **为什么这么慢？** 先看 `jax.devices()`。显示 `CpuDevice` 就是在用 CPU——目前 GB10 上属于正常。
- **连不上 MSA 服务器？** 用 `--msa-mode custom` 传预算好的 A3M，或在本地跑 `colabfold_search`。

## 致谢与许可

- ColabFold：Sergey Ovchinnikov（[@sokrypton](https://twitter.com/sokrypton)）、Milot Mirdita、Martin Steinegger。
- LocalColabFold（上游）：Yoshitaka Moriwaki（[@YoshitakaMo](https://github.com/YoshitakaMo)）。
- 本 fork 只改安装脚本和文档，科学贡献全部归上游。
- 许可：[LICENSE](LICENSE)（继承上游）。

### 引用

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. *ColabFold — Making protein folding accessible to all.* Nature Methods (2022). doi:[10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- 用 **AlphaFold**：Jumper et al. Nature (2021). doi:[10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- 用 **AlphaFold-Multimer**：Evans et al. bioRxiv (2022). doi:[10.1101/2021.10.04.463034](https://doi.org/10.1101/2021.10.04.463034)
