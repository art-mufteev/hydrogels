FROM ubuntu:22.04 AS emc-fetch

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG EMC_URL="https://sourceforge.net/projects/montecarlo/files/emc_linux_x86_64_v9.4.4_20260701.tgz/download"
ARG EMC_SHA256=""

WORKDIR /tmp
RUN wget -O emc.tgz "${EMC_URL}" \
 && if [ -n "${EMC_SHA256}" ]; then \
        echo "${EMC_SHA256}  emc.tgz" | sha256sum -c -; \
    else \
        echo "WARNING: EMC_SHA256 not set, skipping checksum"; \
    fi \
 && mkdir -p /opt/emc \
 && tar -xzf emc.tgz -C /opt/emc --strip-components=1 \
 && rm emc.tgz \
 && test -x /opt/emc/bin/emc_linux_x86_64 \
 && test -d /opt/emc/field \
 && /opt/emc/bin/emc_linux_x86_64 -h 2>&1 | grep -q "EMC: Enhanced Monte Carlo" \
 && echo "=== EMC OK ==="

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS mpi-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
    libnuma-dev libevent-dev libhwloc-dev \
 && rm -rf /var/lib/apt/lists/*

ARG OPENMPI_VERSION=4.1.6
ARG OPENMPI_SHA256=""

WORKDIR /opt
RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.bz2 \
 && if [ -n "${OPENMPI_SHA256}" ]; then \
        echo "${OPENMPI_SHA256}  openmpi-${OPENMPI_VERSION}.tar.bz2" | sha256sum -c -; \
    fi \
 && tar -xjf openmpi-${OPENMPI_VERSION}.tar.bz2 \
 && cd openmpi-${OPENMPI_VERSION} \
 && ./configure --prefix=/opt/openmpi \
                --with-cuda=/usr/local/cuda \
                --with-cuda-libdir=/usr/local/cuda/lib64/stubs \
                --enable-mpi-thread-multiple \
 && make -j$(nproc) \
 && make install

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS lmp-build

COPY --from=mpi-build /opt/openmpi /opt/openmpi
ENV PATH=/opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    libfftw3-dev liblapack-dev libblas-dev \
    libnuma-dev libevent-dev libhwloc-dev \
 && rm -rf /var/lib/apt/lists/*

ARG LMP_TAG=stable_22Jul2025_update5
RUN git clone -b ${LMP_TAG} --depth 1 \
        https://github.com/lammps/lammps.git /opt/lammps \
 && chmod +x /opt/lammps/lib/kokkos/bin/nvcc_wrapper

WORKDIR /opt/lammps/build

RUN cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/opt/lammps/install \
        -D CMAKE_CXX_COMPILER=/opt/lammps/lib/kokkos/bin/nvcc_wrapper \
        -D CMAKE_C_COMPILER=gcc \
        -D CMAKE_CXX_FLAGS="-march=icelake-server -O3" \
        -D BUILD_MPI=ON \
        -D BUILD_OMP=ON \
        -D PKG_KOKKOS=ON \
        -D Kokkos_ENABLE_CUDA=ON \
        -D Kokkos_ARCH_AMPERE80=ON \
        -D Kokkos_ARCH_ICX=ON \
        -D Kokkos_ENABLE_OPENMP=ON \
        -D Kokkos_ENABLE_SERIAL=ON \
        -D FFT_KOKKOS=CUFFT \
        -D PKG_KSPACE=ON \
        -D PKG_CLASS2=ON \
        -D PKG_MOLECULE=ON \
        -D PKG_RIGID=ON \
        -D PKG_COLVARS=ON \
        -D PKG_EXTRA-PAIR=ON \
        -D PKG_EXTRA-FIX=ON \
        -D PKG_MISC=ON \
        .. \
 && make -j$(nproc) \
 && make install

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS python-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir \
    rdkit \
    jinja2 \
    pyyaml \
    numpy

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-3 libgomp1 libnuma1 libevent-2.1-7 libhwloc15 \
    libstdc++6 libopenblas0 \
    libxrender1 libxext6 \
    perl \
    tini \
 && rm -rf /var/lib/apt/lists/*

# OpenMPI + LAMMPS
COPY --from=mpi-build /opt/openmpi /opt/openmpi
COPY --from=lmp-build /opt/lammps/install /opt/lammps

# Python venv ( RDKit)
COPY --from=python-build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN printf '%s\n' '/opt/openmpi/lib' '/opt/lammps/lib' \
        > /etc/ld.so.conf.d/openmpi-lammps.conf \
 && ldconfig

ENV PATH=/opt/openmpi/bin:/opt/lammps/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:/opt/lammps/lib:$LD_LIBRARY_PATH

ENV OMPI_MCA_opal_cuda_support=true
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# EMC
COPY --from=emc-fetch /opt/emc /opt/emc
ENV PATH=/opt/emc/bin:$PATH
ENV EMC_ROOT=/opt/emc

# Pipeline
COPY assets/pipeline /opt/pipeline

RUN printf '%s\n' \
    '#!/bin/bash' \
    'echo "=== GPU ==="' \
    'if command -v nvidia-smi >/dev/null 2>&1; then' \
    '    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv' \
    'else echo "WARNING: nvidia-smi not found"; fi' \
    'echo ""' \
    'echo "=== MPI ==="' \
    'mpirun --version 2>&1 | head -1 || true' \
    'ldd /opt/lammps/bin/lmp 2>/dev/null | grep libmpi || true' \
    'echo ""' \
    'echo "=== LAMMPS packages ==="' \
    'lmp -h 2>&1 | grep -E "KOKKOS|KSPACE|CLASS2|COLVARS|MOLECULE" || true' \
    'echo ""' \
    'echo "=== LAMMPS GPU smoke test ==="' \
    'if command -v nvidia-smi >/dev/null 2>&1; then' \
    '    cat > /tmp/test.in <<EOF' \
    'units lj' \
    'atom_style atomic' \
    'lattice fcc 0.8442' \
    'region box block 0 5 0 5 0 5' \
    'create_box 1 box' \
    'create_atoms 1 box' \
    'mass 1 1.0' \
    'pair_style lj/cut/kk 2.5' \
    'pair_coeff 1 1 1.0 1.0 2.5' \
    'run 0' \
    'EOF' \
    '    lmp -k on g 1 -sf kk -in /tmp/test.in 2>&1 | grep -E "Kokkos|Total wall" | head -3 || echo "smoke test failed"' \
    'else echo "(no GPU — skipping smoke test)"; fi' \
    'echo ""' \
    'echo "=== EMC binary ==="' \
    'emc_linux_x86_64 -h 2>&1 | head -3 || echo "ERROR: EMC failed"' \
    'sha256sum /opt/emc/bin/emc_linux_x86_64' \
    'echo ""' \
    'echo "=== EMC field ==="' \
    'ls /opt/emc/field/cff/pcff/ 2>/dev/null && echo "field OK" || echo "ERROR: field missing"' \
    'echo ""' \
    'echo "=== Perl modules ==="' \
    'perl -e "use File::Path; use Getopt::Long; use File::Basename; use Cwd; print \"Perl modules OK\n\"" || echo "WARNING: Perl modules missing"' \
    'echo ""' \
    'echo "=== Python imports ==="' \
    'python3 -c "import jinja2, yaml, numpy, rdkit; print(\'jinja2, pyyaml, numpy, rdkit -- imports OK\')"' \
    'echo ""' \
    'echo "=== RDKit version ==="' \
    'python3 -c "import rdkit; print(\'rdkit version:\', rdkit.__version__)"' \
    'echo ""' \
    'echo "=== RDKit smoke test (HEA) ==="' \
    'python3 - <<PYEOF' \
    'from rdkit import Chem' \
    'from rdkit.Chem import AllChem, Descriptors' \
    'm = Chem.AddHs(Chem.MolFromSmiles("C=CC(=O)OCCO"))' \
    'print("HEA M =", round(Descriptors.MolWt(m), 2))' \
    'AllChem.EmbedMolecule(m, randomSeed=42)' \
    'AllChem.MMFFOptimizeMolecule(m)' \
    'v = AllChem.ComputeMolVolume(m)' \
    'print("HEA V_w = %.2f A^3" % v)' \
    'PYEOF' \
    'echo ""' \
    'echo "=== numpy BLAS ==="' \
    'python3 -c "import numpy; numpy.show_config()" 2>&1 | grep -iE "openblas|blas" | head -3' \
    'echo ""' \
    'echo "=== Pipeline files ==="' \
    'ls /opt/pipeline/ 2>/dev/null | head -10' \
    > /usr/local/bin/check-all \
 && chmod +x /usr/local/bin/check-all

WORKDIR /work
ENTRYPOINT ["/usr/bin/tini", "--"]