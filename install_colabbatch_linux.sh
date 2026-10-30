#!/bin/bash -e
#
# LocalColabFold installer for Linux — x86_64 and aarch64 (ARM64)
#
#   x86_64 : upstream ColabFold stack, unchanged
#   aarch64: ARM64-native Miniforge + pinned known-good JAX (CPU backend, see README)
#
# Usage:  bash install_colabbatch_linux.sh
# Installs into ./localcolabfold next to the current working directory.
# Do not move that directory after installation.

REPO_RAW="https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main"

type wget 2>/dev/null || { echo "wget is not installed. Please install it using apt or yum." ; exit 1 ; }

ARCH_NAME="$(uname -m)"
case "${ARCH_NAME}" in
    x86_64|amd64)
        echo "Architecture: x86_64"
        MINIFORGE="Miniforge3-Linux-x86_64.sh"
        CONDA_PKGS="git python=3.10 openmm==8.2.0 pdbfixer kalign2=2.04 hhsuite=3.3.0 mmseqs2"
        COLABFOLD_SPEC="colabfold[alphafold-minus-jax]"
        JAX_SPEC="jax[cuda12]==0.5.3"
        NUMPY_SPEC=""
        EXTRA_PIP="tensorflow silence_tensorflow"
        ;;
    aarch64|arm64)
        echo "Architecture: aarch64 (ARM64)"
        MINIFORGE="Miniforge3-Linux-aarch64.sh"
        # conda-forge/bioconda ship all of these for linux-aarch64.
        CONDA_PKGS="git python=3.10 openmm==8.2.0 pdbfixer kalign2=2.04 hhsuite=3.3.0 mmseqs2"
        COLABFOLD_SPEC="colabfold[alphafold-minus-jax]"
        # Pinned together because this is the combination that has been run end-to-end
        # on an NVIDIA DGX Spark (GB10). jax >= 0.7 needs python >= 3.11 and is CPU-less
        # here anyway; the jax_cuda12 plugin cannot see sm_121. See README "GPU acceleration".
        JAX_SPEC="jax==0.6.2 jaxlib==0.6.2"
        NUMPY_SPEC="numpy==2.2.6"
        EXTRA_PIP="tensorflow silence_tensorflow"
        ;;
    *)
        echo "Unsupported architecture: ${ARCH_NAME}"
        exit 1
        ;;
esac

# CUDA environment (harmless when no GPU / no toolkit is present)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
for libdir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" /usr/lib/aarch64-linux-gnu; do
    if [ -d "${libdir}" ]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
    fi
done

if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA toolkit: $(nvcc --version | grep -o 'release [0-9.]*' | head -1)"
else
    echo "Warning: nvcc not found. ColabFold will run on CPU."
fi

CURRENTPATH="$(pwd)"
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

mkdir -p "${COLABFOLDDIR}"
cd "${COLABFOLDDIR}"
wget -q -P . "https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE}"
bash "./${MINIFORGE}" -b -p "${COLABFOLDDIR}/conda"
rm "${MINIFORGE}"

source "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
export PATH="${COLABFOLDDIR}/conda/condabin:${PATH}"
conda update -n base conda -y
conda create -p "${COLABFOLDDIR}/colabfold-conda" -c conda-forge -c bioconda ${CONDA_PKGS} -y
conda activate "${COLABFOLDDIR}/colabfold-conda"

PY="${COLABFOLDDIR}/colabfold-conda/bin/python"
PIP="${COLABFOLDDIR}/colabfold-conda/bin/pip"

# install ColabFold and JAX
"${PIP}" install --no-warn-conflicts "${COLABFOLD_SPEC} @ git+https://github.com/sokrypton/ColabFold"
"${PIP}" install --upgrade pip
# shellcheck disable=SC2086
"${PIP}" install --upgrade ${JAX_SPEC}
if [ -n "${NUMPY_SPEC}" ]; then
    "${PIP}" install "${NUMPY_SPEC}"
fi
# shellcheck disable=SC2086
"${PIP}" install --upgrade ${EXTRA_PIP}

# Download the updater from this fork (not from upstream)
wget -qnc -O "${COLABFOLDDIR}/update_linux.sh" "${REPO_RAW}/update_linux.sh"
chmod +x "${COLABFOLDDIR}/update_linux.sh"

# Drop in the environment helper (written inline so the install never depends on a
# second download succeeding)
cat > "${COLABFOLDDIR}/setup_env.sh" <<'SETUP_ENV_EOF'
#!/bin/bash
# ColabFold environment helper (ARM64 / aarch64 fork).
#
#   source /path/to/localcolabfold/setup_env.sh
#
# Sets CUDA paths (x86_64 and aarch64 layouts), prepends the ColabFold env to PATH,
# and picks a JAX_PLATFORMS value that matches what is actually installed.

COLABFOLDDIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PY="${COLABFOLDDIR}/colabfold-conda/bin/python"
if [ ! -x "${PY}" ]; then
    echo "setup_env.sh: no ColabFold python at ${PY}" >&2
    return 1 2>/dev/null || exit 1
fi

# CUDA: libraries live in /usr/lib/aarch64-linux-gnu on ARM64, not /usr/lib/x86_64-linux-gnu.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
for libdir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib" /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu; do
    if [ -d "${libdir}" ]; then
        export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
    fi
done

export PATH="${COLABFOLDDIR}/colabfold-conda/bin:${PATH}"

# A stale JAX_PLATFORMS=cuda turns every run into a hard failure once the CUDA
# plugin is unusable (e.g. jax_cuda12 on sm_121 / GB10). Only claim CUDA when
# JAX really enumerates a GPU in this environment.
if [ -z "${JAX_PLATFORMS+set}" ]; then
    if "${PY}" -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" >/dev/null 2>&1; then
        export JAX_PLATFORMS=cuda
    else
        export JAX_PLATFORMS=cpu
    fi
fi

echo "ColabFold environment:"
echo "  COLABFOLDDIR : ${COLABFOLDDIR}"
echo "  CUDA_HOME    : ${CUDA_HOME}$( [ -d "${CUDA_HOME}" ] || echo ' (absent)')"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS}"
SETUP_ENV_EOF
chmod +x "${COLABFOLDDIR}/setup_env.sh"

pushd "${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"
# Use 'Agg' for non-GUI backend
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# modify the default params directory
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
# suppress warnings related to tensorflow
sed -i -e "s#from io import StringIO#from io import StringIO\nfrom silence_tensorflow import silence_tensorflow\nsilence_tensorflow()#g" batch.py

# Make the tpu_info import optional (module is not installed on most platforms)
"${PY}" - <<'PYTHON_EOF'
import ast

path = "batch.py"
src = open(path).read()

# 1) optional tpu_info import
if "from tpu_info import device" in src and "    from tpu_info import device" not in src:
    src = src.replace(
        "from tpu_info import device",
        "try:\n    from tpu_info import device\nexcept ImportError:\n    device = None",
    )
    open(path, "w").write(src)
    print("Patched batch.py: optional tpu_info import")

# 2) force float32. The assignment has to sit after the module docstring and after
#    every __future__ import, otherwise batch.py is a SyntaxError.
if "jax.config.update('jax_enable_x64', False)" not in src:
    lines = src.splitlines(keepends=True)
    body = ast.parse(src).body
    idx = 0
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        idx = 1                                    # module docstring
    while (idx < len(body) and isinstance(body[idx], ast.ImportFrom)
           and body[idx].module == "__future__"):
        idx += 1                                   # __future__ imports
    at = body[idx].lineno - 1 if idx < len(body) else len(lines)
    lines.insert(at, "import jax\n")
    lines.insert(at + 1, "jax.config.update('jax_enable_x64', False)  # Use float32 by default\n")
    open(path, "w").writelines(lines)
    print("Patched batch.py: jax_enable_x64 = False")
PYTHON_EOF

"${PY}" -m py_compile batch.py && echo "batch.py compiles"
rm -rf __pycache__
popd

# Download weights
"${PY}" -m colabfold.download

echo "-----------------------------------------"
echo "Installation of ColabFold finished."
echo "Before predicting, load the environment:"
echo -e "\tsource ${COLABFOLDDIR}/setup_env.sh"
echo "Then check that JAX sees a backend:"
echo -e "\tpython -c 'import jax; print(jax.devices())'"
echo "For more details, please run 'colabfold_batch --help'."
