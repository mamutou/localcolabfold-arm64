#!/bin/bash -e
#
# Update an existing LocalColabFold installation (x86_64 and aarch64).
#
#   cd /path/to/localcolabfold
#   wget https://raw.githubusercontent.com/mamutou/localcolabfold-arm64/main/update_linux.sh -O update_linux.sh
#   chmod +x update_linux.sh && ./update_linux.sh .
#
# Reinstalls ColabFold and re-pins the JAX/numpy versions this fork is tested with.

# get absolute path of COLABFOLDDIR
COLABFOLDDIR=$(realpath "${1:-$(dirname "$0")}")

if [ ! -d "$COLABFOLDDIR/colabfold-conda" ]; then
    echo "Error! colabfold-conda directory is not present in $COLABFOLDDIR."
    exit 1
fi

case "$(uname -m)" in
    x86_64|amd64)
        JAX_SPEC="jax[cuda12]==0.5.3"
        NUMPY_SPEC="numpy==2.2.5"
        ;;
    aarch64|arm64)
        # Known-good on DGX Spark / GB10 (CPU backend; see README "GPU acceleration")
        JAX_SPEC="jax==0.6.2 jaxlib==0.6.2"
        NUMPY_SPEC="numpy==2.2.6"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

# activate conda in $COLABFOLDDIR/conda
source "${COLABFOLDDIR}/conda/etc/profile.d/conda.sh"
conda activate "$COLABFOLDDIR/colabfold-conda"

PY="${COLABFOLDDIR}/colabfold-conda/bin/python"
PIP="${COLABFOLDDIR}/colabfold-conda/bin/pip"

# reinstall colabfold and alphafold-colabfold
"${PIP}" install --no-warn-conflicts --upgrade --force-reinstall \
    "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
# shellcheck disable=SC2086
"${PIP}" install --force-reinstall ${JAX_SPEC} "${NUMPY_SPEC}"
"${PIP}" install --upgrade tensorflow silence_tensorflow

# use 'agg' for non-GUI backend
cd "${COLABFOLDDIR}/colabfold-conda/lib/python3.10/site-packages/colabfold"
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# modify the default params directory
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py

# The tpu_info import must stay optional, and the float32 configuration must sit after
# the module docstring and every __future__ import in batch.py — both patches are
# re-applied here because the reinstall above overwrites batch.py.
"${PY}" - <<'PYTHON_EOF'
import ast

path = "batch.py"
src = open(path).read()
if "from tpu_info import device" in src and "    from tpu_info import device" not in src:
    src = src.replace(
        "from tpu_info import device",
        "try:\n    from tpu_info import device\nexcept ImportError:\n    device = None",
    )
    open(path, "w").write(src)
    print("Patched batch.py: optional tpu_info import")

if "jax.config.update('jax_enable_x64', False)" not in src:
    lines = src.splitlines(keepends=True)
    body = ast.parse(src).body
    idx = 0
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        idx = 1
    while (idx < len(body) and isinstance(body[idx], ast.ImportFrom)
           and body[idx].module == "__future__"):
        idx += 1
    at = body[idx].lineno - 1 if idx < len(body) else len(lines)
    lines.insert(at, "import jax\n")
    lines.insert(at + 1, "jax.config.update('jax_enable_x64', False)  # Use float32 by default\n")
    open(path, "w").writelines(lines)
    print("Patched batch.py: jax_enable_x64 = False")
PYTHON_EOF

"${PY}" -m py_compile batch.py && echo "batch.py compiles"
rm -rf __pycache__

# refresh the environment helper
if [ -f "${COLABFOLDDIR}/setup_env.sh" ]; then
    echo "Environment helper left untouched: ${COLABFOLDDIR}/setup_env.sh"
else
    echo "Warning: ${COLABFOLDDIR}/setup_env.sh is missing; re-run install_colabbatch_linux.sh"
fi

echo "Update finished. Reload your shell, then: python -c 'import jax; print(jax.devices())'"
