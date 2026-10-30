# LocalColabFold — ARM64 (aarch64) fork

[ColabFold](https://github.com/sokrypton/ColabFold) on your own **64-bit ARM (aarch64) Linux** machine, with no Colab timeouts and no GPU quota.

This repository is a fork of [YoshitakaMo/localcolabfold](https://github.com/YoshitakaMo/localcolabfold). Upstream's installer is written for x86_64 Linux and macOS: it downloads `Miniforge3-Linux-x86_64.sh`, installs `jax[cuda12]` from x86_64-only wheel naming, and ships Intel/M1 macOS scripts. On an ARM64 server (NVIDIA DGX Spark / GB10, Grace Hopper, Graviton, Ampere Altra + GPU, …) that recipe stops immediately at the conda bootstrap and never reaches a working `colabfold_batch`.

This fork keeps upstream's structure and command-line workflow, but replaces the architecture-specific pieces with ARM64-native ones and documents the aarch64 pitfalls that are otherwise invisible until you hit them.

- Upstream project: <https://github.com/YoshitakaMo/localcolabfold>
- This fork: <https://github.com/mamutou/localcolabfold-arm64>

## Tested environment

This is the machine the installer in this repository was developed against — a single NVIDIA DGX Spark:

| Item | Value |
| --- | --- |
| Machine | NVIDIA DGX Spark, GB10 (Blackwell, compute capability 12.1), 128 GB unified memory |
| OS | Ubuntu 24.04.4 LTS, `aarch64` |
| Driver / CUDA toolkit | 580.173.02 / CUDA 13.0.3 (`/usr/local/cuda-13.0`) |
| Python (ColabFold env) | 3.10.19 (conda-forge, Miniforge3-Linux-aarch64) |
| ColabFold | 1.5.5 |
| JAX | `jax`/`jaxlib` 0.6.2 (+ `jax_cuda12_plugin` 0.6.2, **not usable on sm_121**, see below) |
| Other pins | `numpy` 2.2.6, `openmm` 8.2.0, `tensorflow` 2.20.0 |

Real predictions from this machine (monomer and multimer, `alphafold2_ptm` / `alphafold2_multimer_v3`, default MSA server) completed successfully, so the CPU path is end-to-end verified. The GPU path is **not**: see [GPU acceleration](#gpu-acceleration-experimental).

## What this fork changes

| File | Change |
| --- | --- |
| `install_colabbatch_linux.sh` | Detects `x86_64` vs `aarch64` and picks the matching Miniforge3; on aarch64 installs the ARM64-native toolchain, a pinned known-good JAX, applies the ARM64 patches, and writes `setup_env.sh`. |
| `setup_env.sh` | New. Source it before running `colabfold_batch` to set `CUDA_HOME`, `LD_LIBRARY_PATH` (incl. `/usr/lib/aarch64-linux-gnu`) and a `JAX_PLATFORMS` value that does not break when CUDA is unavailable. |
| Updater download URLs | Upstream installers pulled `update_*.sh` from `YoshitakaMo/localcolabfold`; they now pull from this fork, so a fork install stays on the fork. |
| `README.md` / `README_zh.md` | Rewritten for ARM64 (this file / [中文说明](README_zh.md)). |
| `README_ja.md` | Removed — it was an older Japanese copy of the upstream README and no longer matched the scripts. |
| `beta_install_colabbatch_linux.sh`, `beta_update_linux.sh` | Removed — pre-1.2 legacy scripts pinned to python 3.8/3.9 and to an **x86_64** CUDA-11 `jaxlib` wheel. Nothing in them can work on aarch64, and they were already broken on x86_64. |
| macOS installers | Kept unmodified for convenience; they are Apple-only and irrelevant to ARM64 Linux. Use upstream for Intel/M-series Macs. |

Everything outside these files (`colabfold_batch` behaviour, input formats, flags, model weights) is upstream ColabFold and unchanged.

## Installation (Linux ARM64)

### Prerequisites

1. `curl`, `git`, `wget`:

   ```bash
   sudo apt -y install curl git wget
   ```

2. GNU compiler **12.0 or later** (`GLIBCXX_3.4.30` is required by openmm 8.2 for `--amber` relaxation):

   ```bash
   gcc --version
   ```

   On Ubuntu 24.04 the default `gcc-13` is fine.

3. If you plan to use a GPU, a CUDA **13.x** toolkit matching your driver. Check with `nvcc --version`, **not** `nvidia-smi`:

   ```bash
   $ nvcc --version
   Cuda compilation tools, release 13.0, V13.0.88
   ```

4. Enough disk: the conda environment is ~3 GB, the AlphaFold2/ColabFold weights add ~4 GB, results are extra.

### Install

```bash
mkdir -p ~/work && cd ~/work          # do NOT move this directory afterwards
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/install_colabbatch_linux.sh
bash install_colabbatch_linux.sh
```

About 5–15 minutes later you get `~/work/localcolabfold/` with:

```
localcolabfold/
├── conda/                 # Miniforge3 (aarch64)
├── colabfold-conda/       # python 3.10 env: colabfold, jax, openmm, kalign2, hhsuite, mmseqs2
├── colabfold/             # model weights
├── setup_env.sh           # source this before predicting
└── update_linux.sh
```

Keep the network unblocked and read the log as it goes by. If the install fails, delete the `localcolabfold` directory and re-run — the script is not designed to resume.

### Set up your shell

```bash
source ~/work/localcolabfold/setup_env.sh
```

(`setup_env.sh` is idempotent; add that line to `~/.bashrc` if you want it always on. It exports `PATH`, `CUDA_HOME`/`LD_LIBRARY_PATH` and a safe `JAX_PLATFORMS`.)

### First run

```bash
cat > test.fasta <<'EOF'
>sp|P61823
MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN
LTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPN
CAYKTTQANKHIIVACEGNPYVPVHFDASV
EOF

colabfold_batch test.fasta out/
```

Results (`*_unrelaxed_rank_001_*.pdb`, `*_scores_rank_001_*.json`, `*_plddt.png`, `*_pae.png`) land in `out/`. Add `--templates` and `--amber` when you want templates and AMBER relaxation.

## Verify the install

```bash
source ~/work/localcolabfold/setup_env.sh
python -c "import importlib.metadata as m; print(m.version('colabfold'), m.version('jax'))"
python -c "import jax; print(jax.devices())"
colabfold_batch --help | head -5
```

Expected on this stack: `1.5.5 0.6.2`, and `[CpuDevice(id=0)]` (see next section for why it is not a GPU).

## GPU acceleration (experimental)

**Short version: with the pinned stack above, ColabFold runs on CPU on GB10.** Ask JAX for CUDA in that environment and you get:

```
RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices found for platform CUDA
```

Cause: `jaxlib` 0.6.2 / `jax_cuda12_plugin` 0.6.2 ship cubins for pre-Blackwell SMs. GB10 is `sm_121`, so the plugin loads, finds no matching device, and JAX falls back to — or dies on — CPU. Nothing in `LD_LIBRARY_PATH` fixes this; the limitation is inside the wheel.

The situation improved upstream of ColabFold: JAX publishes `jax[cuda13]` (CUDA 13 plugin + PJRT) with `linux_aarch64` wheels from **0.7.2** onward, and Python wheels for those start at **cp311** (no cp310). On this machine the following was verified to enumerate the GPU:

```bash
# separate env — do not touch the ColabFold env
python3.12 -m venv ~/cf_probe/venv          # Ubuntu 24.04 ships python3.12
~/cf_probe/venv/bin/pip install "jax[cuda13]==0.7.2"
~/cf_probe/venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
# 0.7.2 [CudaDevice(id=0)]
```

`colabfold[alphafold-minus-jax]` (1.6.3) also imports cleanly in that venv. What is **not** verified: an end-to-end prediction under jax 0.7.x — ColabFold 1.5–1.6 was written against the jax 0.4/0.5 API surface, so a real run may still fail in the model code. Treat `jax[cuda13]` as the path to try if you need GPU throughput, and validate with one short sequence before trusting a batch.

To use a GPU once you have a working combination, export `JAX_PLATFORMS=cuda` (or leave it unset and let JAX auto-detect) and check `jax.devices()` before launching a long job.

## Measured capacity on DGX Spark

Synthetic sequences, `--msa-mode single_sequence` and a custom A3M at depth 512, 1 recycle, 1 model, 1 seed, bfloat16, 128 GB unified memory, no relaxation:

| Length | single-sequence | MSA depth 512 |
| --- | --- | --- |
| 1000 | — | ✅ |
| 1500 | ✅ | ✅ |
| 2000 | — | ✅ |
| 2500 | ✅ | ✅ |
| 2800 | ✅ | ✅ |
| 3000 | ✅ | ✅ |
| 3200 | ✅ | ✅ |
| 3400 | ✅ (~48 min) | ✅ |
| 3450 | ❌ OOM | ❌ OOM |

So budget **L ≲ 3400** per prediction on this hardware, and note that the memory ceiling moves with MSA depth, `--num-recycle` and model count.

## Known aarch64 issues (and what this fork does about them)

1. **`jax[cuda12]` cannot see Blackwell GPUs.** See [GPU acceleration](#gpu-acceleration-experimental). The x86_64 installer's `jax[cuda12]==0.5.3` line is *not* a working GPU recipe on GB10.
2. **`ModuleNotFoundError: No module named 'tpu_info'`** when importing `colabfold.batch` on some installs. The installer patches the import into a `try/except ImportError`.
3. **`jax.config.update('jax_enable_x64', False)` placed above `from __future__ import ...` in `batch.py`** makes the module a `SyntaxError`. The patch inserts the float32 configuration *after* every `__future__` import and byte-compiles `batch.py` to confirm.
4. **A stale `JAX_PLATFORMS=cuda`** turns every later run into a hard failure once the CUDA plugin is unusable. `setup_env.sh` only sets `cuda` when a CUDA-capable JAX is actually present, and clears it otherwise.
5. **cuDNN/CUDA libraries on aarch64** live in `/usr/lib/aarch64-linux-gnu`, not `/usr/lib/x86_64-linux-gnu`. `setup_env.sh` adds both.
6. **Package availability on `linux-aarch64`** differs from x86_64. Verified on 2026-10 against conda-forge/bioconda and PyPI:

   | Package | `linux-aarch64` |
   | --- | --- |
   | `openmm` (conda-forge) | ✅ 8.2.x available |
   | `hhsuite` = 3.3.0 (bioconda) | ✅ |
   | `mmseqs2` (bioconda) | ✅ |
   | `kalign2` 2.04 / `kalign` 3.x | ✅ |
   | `jaxlib` / `jax-cuda12-*` | ✅ (cp310+) |
   | `jax-cuda13-*` | ✅ from 0.7.2, but **cp311+ only** |
   | `colabfold` (PyPI sdist) | ✅ installed from git/pip |

## Input formats

`colabfold_batch` accepts a FASTA file, an A3M file, a CSV/TSV file, or a directory of FASTA/A3M files.

```fasta
>sp|P61823
MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN
LTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPN
CAYKTTQANKHIIVACEGNPYVPVHFDASV
```

Multimers: separate chains with `:`, and note that numbers in the header are parsed as stoichiometry (`>2x3_A:3_B` style copies also work):

```fasta
>3KUD_RasRaf_complex
MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQ
YMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIP
YIETSAKTRQGVEDAFYTLVREIRQH:
PSKTSNTIRVFLPNKQRTVVNVRNGMSLHDCLMKALKVRGLQPECCAVFRLLHEHKGKKARLDWNTDAAS
LIGEELQVDFL
```

CSV:

```csv
id,sequence
5AWL_1,YYDPETGTWY
```

## Flags worth knowing

| Flag | Effect |
| --- | --- |
| `--amber` | AMBER relaxation; `--num-relax N` limits how many top models get relaxed |
| `--use-gpu-relax` | Run AMBER on GPU (NVIDIA + CUDA only) |
| `--templates` | Use PDB templates; `--custom-template-path DIR` restricts it to local files |
| `--num-recycle N` | More recycles, better/slower; default 3 |
| `--num-seeds N` / `--random-seed N` | Sample several seeds — useful on a CPU-only box where you trade time for diversity |
| `--max-msa 512:1024` | Cap MSA depth — **the cheapest way to fit long sequences on this machine** |
| `--use-dropout` | Sample from model uncertainty |
| `--overwrite-existing-results` | Re-run existing ids |

Full list: `colabfold_batch --help`.

## How to update

ColabFold itself moves fast. Update the environment (not the install directory) with:

```bash
cd ~/work/localcolabfold
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/update_linux.sh -O update_linux.sh
chmod +x update_linux.sh
./update_linux.sh .
```

This reinstalls ColabFold and re-pins the JAX/numpy versions this fork is tested with. If you deliberately upgraded JAX for the GPU path, skip the updater or edit the pins first.

To update this fork's scripts:

```bash
cd ~/work
wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/install_colabbatch_linux.sh -O /tmp/new_installer.sh
diff install_colabbatch_linux.sh /tmp/new_installer.sh
```

## FAQ

- **Do I need the big AlphaFold databases (PDB70, BFD, Uniclust30, MGnify)?** No. MSA generation goes through the MMseqs2 web server, exactly like ColabFold. For offline use, run `setup_databases.sh` from the ColabFold repo and `colabfold_search` locally.
- **Can I use more than one GPU?** No — AlphaFold/ColabFold predict on a single device. With several GPUs, pick one per process with `CUDA_VISIBLE_DEVICES`.
- **Is Apple Silicon (M1/M2/M3) supported here?** Not by this fork; that is upstream's `install_colabbatch_M1mac.sh`, which is left unmodified in this repo. Note its default stack (`jax==0.4.23`) is old.
- **Does it work inside WSL2 on ARM64 Windows?** Untested. Upstream's WSL2 notes (case-sensitive directory setup) still apply.
- **Why is my prediction so slow?** Check `jax.devices()`. If it says `CpuDevice`, you are on CPU — expected on GB10 today.
- **MSA server unreachable?** Pass a precomputed A3M with `--msa-mode custom`/a `.a3m` input, or run `colabfold_search` locally.

## Credits and license

- ColabFold: Sergey Ovchinnikov ([@sokrypton](https://twitter.com/sokrypton)), Milot Mirdita, Martin Steinegger.
- LocalColabFold (upstream): Yoshitaka Moriwaki ([@YoshitakaMo](https://github.com/YoshitakaMo)).
- This fork only changes installers and documentation; all scientific credit belongs upstream.
- License: see [LICENSE](LICENSE) (inherited from upstream).

### Citations

- Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. *ColabFold — Making protein folding accessible to all.* Nature Methods (2022). doi:[10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- Using **AlphaFold**: Jumper et al. Nature (2021). doi:[10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- Using **AlphaFold-Multimer**: Evans et al. bioRxiv (2022). doi:[10.1101/2021.10.04.463034](https://doi.org/10.1101/2021.10.04.463034)
