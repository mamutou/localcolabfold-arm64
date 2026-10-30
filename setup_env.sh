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
