# pcieburn build.
#
# Authored on a machine with no GPU and no CUDA/NCCL, so nothing here is
# hardcoded from a local probe. Everything is overridable and `make preflight`
# prints what it actually resolved before you commit to a build.
#
# Target node for this investigation: driver 580.178.04, CUDA 13.0,
# RTX 5090 (compute capability 12.0 -> sm_120).
#
#   make preflight          # show resolved toolchain, no compile
#   make                    # build ./pcieburn
#   make NCCL_HOME=/opt/nccl
#   make CUDA_HOME=/usr/local/cuda-13.0
#   make SM=90              # different GPU generation

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

# RTX 5090 is compute capability 12.0. CUDA 13 supports sm_120 natively;
# the compute_120 PTX fallback lets the same binary JIT onto a newer part.
SM           ?= 120
NVCC_GENCODE ?= -gencode=arch=compute_$(SM),code=sm_$(SM) \
                -gencode=arch=compute_$(SM),code=compute_$(SM)

# NCCL discovery. Checked in order: an explicit NCCL_HOME, then the CUDA tree,
# then the usual distro locations for libnccl-dev. If none of these hit,
# `make preflight` says so plainly instead of failing deep inside a compile.
NCCL_HOME ?= $(shell \
	for d in $(CUDA_HOME) /usr /usr/local /usr/local/nccl /opt/nccl; do \
		if [ -f "$$d/include/nccl.h" ]; then echo "$$d"; exit 0; fi; \
	done; \
	echo "NCCL_NOT_FOUND")

# CUDA 13 requires C++17; earlier toolkits are fine with it too.
CUDA_MAJOR := $(shell $(NVCC) --version 2>/dev/null | \
	sed -n 's/^.*release \([0-9][0-9]*\)\..*$$/\1/p' | head -1)
CXXSTD     := -std=c++17

INCLUDES := -I$(CUDA_HOME)/include
ifneq ($(NCCL_HOME),NCCL_NOT_FOUND)
ifneq ($(NCCL_HOME),$(CUDA_HOME))
INCLUDES += -I$(NCCL_HOME)/include
endif
endif

# NVCCFLAGS_EXTRA is the supported way to pass one-off defines without editing
# this file, e.g. if a future toolkit drops the deprecated CUBLAS_TENSOR_OP_MATH:
#   make NVCCFLAGS_EXTRA=-DPCIEBURN_TENSOR_MATH=CUBLAS_DEFAULT_MATH
NVCCFLAGS_EXTRA ?=

NVCCFLAGS := $(NVCC_GENCODE) $(CXXSTD) -O3 -lineinfo \
             -Xcompiler -Wall -Xcompiler -Wextra \
             -Xcompiler -Wno-unused-parameter \
             $(INCLUDES) $(NVCCFLAGS_EXTRA)

# Where libnccl.so actually lives. On Debian/Ubuntu the package puts it in the
# multiarch dir, not $(NCCL_HOME)/lib, so search for the real thing rather than
# assuming a layout. Override directly if the search misses:
#   make NCCL_LIBDIR=/usr/lib/x86_64-linux-gnu
NCCL_LIBDIR ?= $(shell \
	for d in $(NCCL_HOME)/lib/$$(uname -m)-linux-gnu $(NCCL_HOME)/lib64 \
	         $(NCCL_HOME)/lib /usr/lib/$$(uname -m)-linux-gnu; do \
		if ls "$$d"/libnccl.so* >/dev/null 2>&1; then echo "$$d"; exit 0; fi; \
	done; \
	echo "")

LDFLAGS := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs
ifneq ($(NCCL_LIBDIR),)
LDFLAGS += -L$(NCCL_LIBDIR) -Xlinker -rpath -Xlinker $(NCCL_LIBDIR)
endif
# No -lrt: everything used here (gettimeofday, usleep) is in libc on modern
# glibc, and linking it only produced an nvlink warning about skipping librt.a.
LDFLAGS += -lnccl -lcublas -lcudart

TARGET := pcieburn

.PHONY: all preflight preflight-quiet clean

all: $(TARGET)

# Order-only prerequisite (after the |). Plain prerequisites of `all` are
# unordered, so under `make -j` nvcc could start before the preflight guard had
# a chance to fail with its actionable message.
$(TARGET): pcieburn.cu | preflight-quiet
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
	@echo "built ./$(TARGET) for sm_$(SM)"

# Fails the build early with an actionable message rather than a wall of
# missing-header errors.
preflight-quiet:
	@if [ ! -x "$(NVCC)" ]; then \
		echo "ERROR: nvcc not found at $(NVCC)"; \
		echo "       set CUDA_HOME=/path/to/cuda"; exit 1; fi
	@if [ "$(NCCL_HOME)" = "NCCL_NOT_FOUND" ]; then \
		echo "ERROR: nccl.h not found in CUDA_HOME or any standard location."; \
		echo "       install libnccl-dev, or set NCCL_HOME=/path/to/nccl"; \
		exit 1; fi

preflight:
	@echo "=== pcieburn preflight ==="
	@echo "CUDA_HOME    : $(CUDA_HOME)"
	@echo "nvcc         : $(NVCC)"
	@if [ -x "$(NVCC)" ]; then \
		echo "nvcc version : $$($(NVCC) --version | tail -1)"; \
	else echo "nvcc version : NOT FOUND"; fi
	@echo "CUDA major   : $(CUDA_MAJOR)"
	@echo "NCCL_HOME    : $(NCCL_HOME)"
	@if [ "$(NCCL_HOME)" != "NCCL_NOT_FOUND" ] && \
	    [ -f "$(NCCL_HOME)/include/nccl.h" ]; then \
		echo "NCCL version : $$(sed -n \
			's/^#define NCCL_MAJOR \([0-9]*\).*/\1/p;' \
			$(NCCL_HOME)/include/nccl.h | head -1).$$(sed -n \
			's/^#define NCCL_MINOR \([0-9]*\).*/\1/p' \
			$(NCCL_HOME)/include/nccl.h | head -1).$$(sed -n \
			's/^#define NCCL_PATCH \([0-9]*\).*/\1/p' \
			$(NCCL_HOME)/include/nccl.h | head -1)"; \
	else echo "NCCL version : header NOT FOUND"; fi
	@if [ -n "$(NCCL_LIBDIR)" ]; then \
		echo "NCCL libdir  : $(NCCL_LIBDIR)"; \
		echo "NCCL so      : $$(ls $(NCCL_LIBDIR)/libnccl.so* 2>/dev/null | tr '\n' ' ')"; \
	else \
		echo "NCCL libdir  : not located — relying on the linker default path"; \
		echo "               (override with NCCL_LIBDIR=... if -lnccl fails)"; fi
	@echo "target arch  : sm_$(SM) (+ compute_$(SM) PTX)"
	@echo "C++ standard : $(CXXSTD)"
	@if [ -f /proc/driver/nvidia/version ]; then \
		echo "driver       : $$(head -1 /proc/driver/nvidia/version)"; \
	else echo "driver       : /proc/driver/nvidia/version absent (no GPU here?)"; fi
	@echo "link flags   : $(LDFLAGS)"

clean:
	$(RM) $(TARGET)
