// pcieburn — combined compute + PCIe/NCCL stress test (design Option B).
//
// One OS process per GPU. Each rank interleaves an unpaced cuBLAS GEMM burst
// (lifted from DCGM's GpuBurnWorker::Compute() / gpu-burn's compute()) with a
// real NCCL collective, so the bus sees the compute-then-allreduce shape that
// tensor-parallel inference actually produces. Neither DCGM's `diagnostic` nor
// nccl-tests does both, which is the gap this exists to close.
//
// Provenance of the reproduced-fault parameters (see README for detail):
//   DCGM nvvs/plugin_src/diagnostic/DiagnosticPlugin.cpp  (Apache-2.0)
//   gpu-burn gpu_burn-drv.cpp / compare.cu                (BSD-2-Clause)
//
// Deliberately NOT using MPI: nccl-tests needs it for multi-process, but a
// fork + pipe rendezvous for the ncclUniqueId is enough for a single node and
// keeps the launch path dependency-free.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <nccl.h>

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

// cublasSetMathMode(CUBLAS_TENSOR_OP_MATH) is what DCGM uses to pull in the
// tensor cores partway through a run. The enumerator has been deprecated since
// CUDA 11; if a future toolkit removes it outright, rebuild with
// -DPCIEBURN_TENSOR_MATH=CUBLAS_DEFAULT_MATH rather than editing this file.
#ifndef PCIEBURN_TENSOR_MATH
#define PCIEBURN_TENSOR_MATH CUBLAS_TENSOR_OP_MATH
#endif

// ---------------------------------------------------------------------------
// Timestamps. Format matches bmc_power_poll.py exactly (UTC, milliseconds,
// trailing Z) so the harness log and the BMC/NVML traces join without any
// reformatting step.
// ---------------------------------------------------------------------------

static void isoNow(char *out, size_t n)
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    struct tm tmv;
    gmtime_r(&tv.tv_sec, &tmv);
    // base is sized to just fit "YYYY-MM-DDTHH:MM:SS" (19 chars + NUL) and the
    // millisecond field is clamped to three digits, so the composed string
    // provably fits the callers' 48-byte buffers with no truncation.
    char base[24];
    strftime(base, sizeof(base), "%Y-%m-%dT%H:%M:%S", &tmv);
    int ms = (int)(tv.tv_usec / 1000);
    if (ms < 0)
        ms = 0;
    if (ms > 999)
        ms = 999;
    snprintf(out, n, "%s.%03dZ", base, ms);
}

static double nowSec()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// Role prefix so interleaved rank/supervisor output stays readable.
static int g_logRank = -1;

static void logf_(const char *fmt, ...)
{
    char ts[48];
    isoNow(ts, sizeof(ts));
    char role[32];
    if (g_logRank < 0)
        snprintf(role, sizeof(role), "sup");
    else
        snprintf(role, sizeof(role), "rank%d", g_logRank);

    char body[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    // Compose the whole line and emit it with ONE write(). All ranks share this
    // fd, and fprintf on an unbuffered stream is not guaranteed to issue a
    // single syscall — the first smoke run tore lines apart mid-field
    // ("...nan=" on one line, "0" on the next). A torn line at the instant of a
    // fault would corrupt the one artifact that timestamps it. A single write
    // under PIPE_BUF is atomic with respect to other writers, and the fd is
    // inherited through fork so the file offset is shared.
    char line[1280];
    int n = snprintf(line, sizeof(line), "[%s] %-6s %s\n", ts, role, body);
    if (n < 0)
        return;
    if ((size_t)n > sizeof(line) - 1)
        n = (int)(sizeof(line) - 1);
    ssize_t ignored = write(STDERR_FILENO, line, (size_t)n);
    (void)ignored;
}

// ---------------------------------------------------------------------------
// Error handling. Ranks must never quietly continue past a failure: a rank that
// stops participating in collectives hangs every other rank, so any error has
// to become a fast, loud process exit that the supervisor can see.
// ---------------------------------------------------------------------------

#define RANK_DIE(...)                     \
    do {                                  \
        logf_(__VA_ARGS__);               \
        rankAbort();                      \
        _exit(EXIT_RANK_FAILURE);         \
    } while (0)

#define CUDA_CHECK(cmd)                                                        \
    do {                                                                       \
        cudaError_t e_ = (cmd);                                                \
        if (e_ != cudaSuccess)                                                 \
            RANK_DIE("CUDA error %s:%d '%s' -> %s", __FILE__, __LINE__, #cmd,  \
                     cudaGetErrorString(e_));                                  \
    } while (0)

#define CUBLAS_CHECK(cmd)                                                      \
    do {                                                                       \
        cublasStatus_t s_ = (cmd);                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                       \
            RANK_DIE("cuBLAS error %s:%d '%s' -> status %d", __FILE__,         \
                     __LINE__, #cmd, (int)s_);                                 \
    } while (0)

#define NCCL_CHECK(cmd)                                                        \
    do {                                                                       \
        ncclResult_t r_ = (cmd);                                               \
        if (r_ != ncclSuccess)                                                 \
            RANK_DIE("NCCL error %s:%d '%s' -> %s", __FILE__, __LINE__, #cmd,  \
                     ncclGetErrorString(r_));                                  \
    } while (0)

static const int EXIT_RANK_FAILURE = 42;
static const int EXIT_RANK_HANG    = 43;

// Set once the comm exists so RANK_DIE can tear it down instead of leaving
// peers blocked in a collective that will never complete.
static ncclComm_t g_comm     = nullptr;
static bool       g_commLive = false;

static void rankAbort()
{
    if (g_commLive && g_comm) {
        g_commLive = false;
        ncclCommAbort(g_comm);
    }
}

// ---------------------------------------------------------------------------
// Compare kernels. Reimplemented from DCGM's compare.cu (grid-stride, warp
// shuffle reduction, one atomic per warp) rather than loaded from a fatbin:
// gpu-burn's runtime cuModuleLoad path uses the pre-CUDA-4.0 launch API
// (cuParamSetv/cuLaunchGridAsync), which is gone in CUDA 13.
//
// Every C_i holds A*B, so C_i must equal C_0 within epsilon. A mismatch is a
// real compute fault; a NaN usually means the GEMM ran on wedged hardware.
// ---------------------------------------------------------------------------

#define PB_EPSILON_F 0.001f
#define PB_EPSILON_D 0.0000001

__device__ static void reduceThenAtomicAdd(int val, int *out)
{
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, off);
    if ((threadIdx.x & 31) == 0 && val != 0)
        atomicAdd(out, val);
}

__device__ __forceinline__ static size_t flatIndex()
{
    return (size_t)(blockIdx.y * blockDim.y + threadIdx.y) *
               (size_t)(gridDim.x * blockDim.x) +
           (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ __forceinline__ static size_t flatStride()
{
    return (size_t)blockDim.x * blockDim.y * gridDim.x * gridDim.y;
}

extern "C" __global__ void compareFP32(float **C, int *faulty, int *nans,
                                       size_t iters, size_t nElems)
{
    int f = 0, n = 0;
    const size_t stride = flatStride();
    for (size_t idx = flatIndex(); idx < nElems; idx += stride) {
        const float ref = C[0][idx];
        if (isnan(ref))
            n++;
        for (size_t i = 1; i < iters; i++) {
            const float v = C[i][idx];
            if (isnan(v))
                n++;
            else if (fabsf(ref - v) > PB_EPSILON_F)
                f++;
        }
    }
    reduceThenAtomicAdd(f, faulty);
    reduceThenAtomicAdd(n, nans);
}

extern "C" __global__ void compareFP64(double **C, int *faulty, int *nans,
                                       size_t iters, size_t nElems)
{
    int f = 0, n = 0;
    const size_t stride = flatStride();
    for (size_t idx = flatIndex(); idx < nElems; idx += stride) {
        const double ref = C[0][idx];
        if (isnan(ref))
            n++;
        for (size_t i = 1; i < iters; i++) {
            const double v = C[i][idx];
            if (isnan(v))
                n++;
            else if (fabs(ref - v) > PB_EPSILON_D)
                f++;
        }
    }
    reduceThenAtomicAdd(f, faulty);
    reduceThenAtomicAdd(n, nans);
}

extern "C" __global__ void compareFP16(__half **C, int *faulty, int *nans,
                                       size_t iters, size_t nElems)
{
    int f = 0, n = 0;
    const size_t stride = flatStride();
    for (size_t idx = flatIndex(); idx < nElems; idx += stride) {
        const float ref = __half2float(C[0][idx]);
        if (isnan(ref))
            n++;
        for (size_t i = 1; i < iters; i++) {
            const float v = __half2float(C[i][idx]);
            if (isnan(v))
                n++;
            else if (fabsf(ref - v) > PB_EPSILON_F)
                f++;
        }
    }
    reduceThenAtomicAdd(f, faulty);
    reduceThenAtomicAdd(n, nans);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum Precision { PREC_HALF = 0, PREC_SINGLE = 1, PREC_DOUBLE = 2 };

static const char *precName(Precision p)
{
    switch (p) {
    case PREC_HALF:   return "half";
    case PREC_SINGLE: return "single";
    default:          return "double";
    }
}

enum Collective { COLL_ALLREDUCE = 0, COLL_ALLTOALL = 1, COLL_SENDRECV = 2 };

static const char *collName(Collective c)
{
    switch (c) {
    case COLL_ALLREDUCE: return "allreduce";
    case COLL_ALLTOALL:  return "alltoall";
    default:             return "sendrecv";
    }
}

// How many bytes actually leave a rank per byte of collective buffer. These are
// the same factors nccl-tests uses for its busBw column (all_reduce.cu:59-65,
// sendrecv.cu:38-44), so the numbers here are directly comparable to a
// nccl-tests run on the same fabric.
//
//   allreduce : ring moves 2(N-1)/N in each direction
//   alltoall  : the 1/N destined for self never leaves the GPU
//   sendrecv  : the whole buffer goes to exactly one peer
static double collBwFactor(Collective c, int nranks)
{
    if (nranks < 2)
        return 0.0;
    const double n = (double)nranks;
    switch (c) {
    case COLL_ALLREDUCE: return 2.0 * (n - 1.0) / n;
    case COLL_ALLTOALL:  return (n - 1.0) / n;
    default:             return 1.0;
    }
}

// This fleet has PCIe P2P disabled entirely (every pair reports CNS), so NCCL
// stages every transfer through host RAM. Each remote byte therefore crosses
// PCIe twice — up the sender's link into host memory, then down the receiver's.
// For one rank that means its own link carries both its egress and its ingress.
static const double PCIE_HOST_STAGING_MULT = 2.0;

// PCIe Gen5 x16: 32 GT/s per lane x 16 lanes x 128b/130b ~= 63 GB/s per
// direction. Used only to express an achieved rate as a fraction of what the
// link can do; it is a spec figure, not a measurement.
static const double PCIE_GEN5_X16_GBPS = 63.0;

struct Config {
    double duration       = 60.0;
    unsigned matrixDim    = 2048;      // DCGM DIAGNOSTIC_STR_MATRIX_DIM default
    std::vector<Precision> precisions; // default {half, single}
    size_t gemmsPerColl   = 0;         // 0 = full DCGM-style burst per collective
    Collective collective = COLL_ALLREDUCE;
    size_t collMin        = 128ull << 20;
    size_t collMax        = 1ull << 30;
    unsigned collFactor   = 2;
    double memFrac        = 0.9;       // DCGM/gpu-burn USEMEM
    size_t maxCBuffers    = 0;         // 0 = unlimited
    bool explicitStream   = false;     // default: legacy NULL stream, as DCGM
    bool alwaysTensor     = false;
    bool doCompare        = true;
    double collTimeout    = 120.0;     // in-rank stream poll timeout, 0 = off
    double watchdog       = 60.0;      // supervisor silence timeout, 0 = off
    double reportInterval = 1.0;
    std::string eventLog;
    std::string tag;
    std::vector<int> gpus;             // empty = all visible devices
};

// ---------------------------------------------------------------------------
// Rank <-> supervisor wire protocol. Fixed-size struct, single writer per pipe,
// well under PIPE_BUF, so each write is atomic and never interleaves.
// ---------------------------------------------------------------------------

static const uint32_t MSG_MAGIC = 0x50427531u; // "PBu1"

enum MsgKind { MSG_READY = 0, MSG_PROGRESS = 1, MSG_DONE = 2, MSG_ERROR = 3 };

struct RankMsg {
    uint32_t magic;
    int32_t  rank;
    int32_t  kind;
    int32_t  precision;
    int64_t  outerIter;
    int64_t  gemms;      // cumulative
    int64_t  colls;      // cumulative
    int64_t  collBytes;  // cumulative
    int64_t  faulty;     // cumulative
    int64_t  nans;       // cumulative
    double   wall;       // seconds since epoch
    double   gflops;     // most recent burst
    char     text[96];
};

static void msgInit(RankMsg *m, int rank, MsgKind kind)
{
    memset(m, 0, sizeof(*m));
    m->magic = MSG_MAGIC;
    m->rank  = rank;
    m->kind  = (int32_t)kind;
    m->wall  = nowSec();
}

static bool writeAll(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        p += w;
        n -= (size_t)w;
    }
    return true;
}

static bool readAll(int fd, void *buf, size_t n)
{
    char *p = (char *)buf;
    while (n) {
        ssize_t r = read(fd, p, n);
        if (r == 0)
            return false; // EOF
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        p += r;
        n -= (size_t)r;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Rank-side state
// ---------------------------------------------------------------------------

static volatile sig_atomic_t g_stopRequested = 0;

static void onTerm(int) { g_stopRequested = 1; }

struct DeviceBuffers {
    // A and B are allocated once per precision and never change; every GEMM
    // multiplies the same A*B and only the destination C advances. Matches
    // DCGM and gpu-burn.
    void *aFP16 = nullptr, *bFP16 = nullptr;
    void *aFP32 = nullptr, *bFP32 = nullptr;
    void *aFP64 = nullptr, *bFP64 = nullptr;

    // baseIterations buffers of FP64 matrix size. FP32 aliases 2 matrices into
    // each, FP16 aliases 4 — exactly DCGM's InitBuffers() layout, so the memory
    // footprint and per-precision GEMM counts match the reproducing run.
    std::vector<void *> base;
    std::vector<void *> cHost[3];   // host-side pointer lists, per precision
    void *cDev[3] = {nullptr, nullptr, nullptr}; // device arrays of pointers
    size_t iters[3] = {0, 0, 0};

    int *faultyDev = nullptr;
    int *nanDev    = nullptr;

    void *collSend = nullptr;
    void *collRecv = nullptr;
    int  *stopDev  = nullptr;
    unsigned long long *reduceDev = nullptr;

    // Pinned staging for every host<->device scalar transfer. This is load
    // bearing, not an optimization: cudaMemcpyAsync to or from PAGEABLE host
    // memory is synchronous with respect to the host, so a pageable readback
    // blocks inside the copy until the entire enqueued stream has drained.
    // Control would never reach pollStream, --coll-timeout would never arm, and
    // a GPU that fell off the bus would wedge the process in cudaMemcpyAsync
    // instead of producing a timestamped line — the exact failure this design
    // is built to avoid.
    int *stageInt = nullptr;   // [0]=faulty [1]=nans [2]=wantStop [3]=agreedStop
    unsigned long long *stageU64 = nullptr;

    // Bracket the GEMM window so reported throughput is GPU time for the GEMMs,
    // not wall time for the whole pass (which would include collectives).
    cudaEvent_t evGemmStart = nullptr;
    cudaEvent_t evGemmEnd   = nullptr;
};

// Poll for stream completion instead of blocking in cudaStreamSynchronize.
// Adapted from nccl-tests testStreamSynchronize (src/common.cu:482). This is
// the whole reason a fall-off-the-bus event produces a timestamped line rather
// than a wedged process: cudaStreamSynchronize on a dead GPU never returns.
static bool pollStream(cudaStream_t stream, ncclComm_t comm, double timeout,
                       const char **whyOut)
{
    const double t0 = nowSec();
    for (;;) {
        cudaError_t q = cudaStreamQuery(stream);
        if (q == cudaSuccess)
            return true;
        if (q != cudaErrorNotReady) {
            *whyOut = cudaGetErrorString(q);
            return false;
        }

        if (comm) {
            ncclResult_t async = ncclSuccess;
            ncclResult_t got   = ncclCommGetAsyncError(comm, &async);
            if (got != ncclSuccess) {
                *whyOut = ncclGetErrorString(got);
                return false;
            }
            if (async != ncclSuccess) {
                *whyOut = ncclGetErrorString(async);
                return false;
            }
        }

        if (timeout > 0.0 && (nowSec() - t0) > timeout) {
            *whyOut = "stream/collective timeout — suspected hang";
            return false;
        }

        // Yielding rather than spinning hot keeps the host from competing with
        // NCCL's own progress threads for CPU.
        usleep(200);
    }
}

// GFLOP per GEMM, from DCGM's CalculateGFlopsMultiplier (OPS_PER_2048_MUL
// scaled by (dim/2048)^3), so throughput numbers are comparable to a dcgmi
// diag run on the same node.
static double gflopsPerGemm(unsigned dim)
{
    const double opsPer2048 = 17188257792.0;
    const double scale      = (double)dim / 2048.0;
    return opsPer2048 * scale * scale * scale / 1073741824.0;
}

static size_t collSizeFor(const Config &cfg, long long iter)
{
    // Cycle deterministically through the sweep. Every rank derives the size
    // from the same iteration number, so all ranks always agree.
    std::vector<size_t> sizes;
    for (size_t s = cfg.collMin; s <= cfg.collMax;
         s = (cfg.collFactor > 1) ? s * cfg.collFactor : s + cfg.collMin) {
        sizes.push_back(s);
        if (cfg.collFactor <= 1 && sizes.size() > 64)
            break;
    }
    if (sizes.empty())
        sizes.push_back(cfg.collMin);
    return sizes[(size_t)(iter % (long long)sizes.size())];
}

static size_t collSizeCount(const Config &cfg)
{
    size_t n = 0;
    for (size_t s = cfg.collMin; s <= cfg.collMax;
         s = (cfg.collFactor > 1) ? s * cfg.collFactor : s + cfg.collMin) {
        n++;
        if (cfg.collFactor <= 1 && n > 64)
            break;
    }
    return n ? n : 1;
}

// Issue one collective of `bytes` on `stream`. Nothing here synchronizes — the
// collective is queued behind the GEMM burst on the same stream, which is what
// produces the compute-then-communicate ordering of a real transformer layer.
static void issueCollective(const Config &cfg, DeviceBuffers &db, ncclComm_t comm,
                            cudaStream_t stream, int rank, int nranks,
                            size_t bytes)
{
    const size_t count = bytes / sizeof(float);
    if (count == 0)
        return;

    switch (cfg.collective) {
    case COLL_ALLREDUCE:
        NCCL_CHECK(ncclAllReduce(db.collSend, db.collRecv, count, ncclFloat,
                                 ncclSum, comm, stream));
        break;

    case COLL_ALLTOALL: {
        const size_t per = count / (size_t)nranks;
        if (per == 0)
            return;
        NCCL_CHECK(ncclGroupStart());
        for (int p = 0; p < nranks; p++) {
            NCCL_CHECK(ncclSend((char *)db.collSend + (size_t)p * per * sizeof(float),
                                per, ncclFloat, p, comm, stream));
            NCCL_CHECK(ncclRecv((char *)db.collRecv + (size_t)p * per * sizeof(float),
                                per, ncclFloat, p, comm, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        break;
    }

    case COLL_SENDRECV: {
        const int next = (rank + 1) % nranks;
        const int prev = (rank - 1 + nranks) % nranks;
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(db.collSend, count, ncclFloat, next, comm, stream));
        NCCL_CHECK(ncclRecv(db.collRecv, count, ncclFloat, prev, comm, stream));
        NCCL_CHECK(ncclGroupEnd());
        break;
    }
    }
}

// The GEMM burst. Verbatim in shape from DiagnosticPlugin.cpp:1249 — same
// CUBLAS_OP_N/CUBLAS_OP_N, m=n=k=lda=ldb=ldc=matrixDim, alpha=1, beta=0 in the
// native type, one GEMM per C buffer, and crucially NO synchronization between
// launches. That gapless pattern is the one thing every real reproduction of
// this fault has had in common, so it is preserved exactly.
static void gemmBurst(const Config &cfg, DeviceBuffers &db, cublasHandle_t cub,
                      Precision prec, size_t from, size_t to)
{
    static const float  alphaF = 1.0f,  betaF = 0.0f;
    static const double alphaD = 1.0,   betaD = 0.0;
    const __half        alphaH = __float2half(1.0f);
    const __half        betaH  = __float2half(0.0f);

    const int dim = (int)cfg.matrixDim;

    for (size_t i = from; i < to; i++) {
        if (prec == PREC_HALF) {
            CUBLAS_CHECK(cublasHgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                     &alphaH, (const __half *)db.aFP16, dim,
                                     (const __half *)db.bFP16, dim, &betaH,
                                     (__half *)db.cHost[PREC_HALF][i], dim));
        } else if (prec == PREC_SINGLE) {
            CUBLAS_CHECK(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                     &alphaF, (const float *)db.aFP32, dim,
                                     (const float *)db.bFP32, dim, &betaF,
                                     (float *)db.cHost[PREC_SINGLE][i], dim));
        } else {
            CUBLAS_CHECK(cublasDgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim,
                                     &alphaD, (const double *)db.aFP64, dim,
                                     (const double *)db.bFP64, dim, &betaD,
                                     (double *)db.cHost[PREC_DOUBLE][i], dim));
        }
    }
}

static void runCompare(const Config &cfg, DeviceBuffers &db, Precision prec,
                       cudaStream_t stream)
{
    const size_t iters  = db.iters[prec];
    const size_t nElems = (size_t)cfg.matrixDim * cfg.matrixDim;

    // Zero the counters BEFORE the early return. They are cudaMalloc'd and
    // otherwise never initialized, so bailing out first would leave the caller
    // reading uninitialized device memory as a fault count — fabricating a
    // compute-fault verdict, which is worse than reporting nothing.
    CUDA_CHECK(cudaMemsetAsync(db.faultyDev, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(db.nanDev, 0, sizeof(int), stream));

    // Fewer than two output matrices means there is nothing to compare against.
    if (iters < 2)
        return;

    // DCGM's geometry: grid (64,64), block (32,8).
    dim3 grid(64, 64, 1), block(32, 8, 1);
    if (prec == PREC_HALF)
        compareFP16<<<grid, block, 0, stream>>>((__half **)db.cDev[prec],
                                                db.faultyDev, db.nanDev, iters,
                                                nElems);
    else if (prec == PREC_SINGLE)
        compareFP32<<<grid, block, 0, stream>>>((float **)db.cDev[prec],
                                                db.faultyDev, db.nanDev, iters,
                                                nElems);
    else
        compareFP64<<<grid, block, 0, stream>>>((double **)db.cDev[prec],
                                                db.faultyDev, db.nanDev, iters,
                                                nElems);
}

// ---------------------------------------------------------------------------
// The rank worker. Runs in a forked child; never returns.
// ---------------------------------------------------------------------------

static void runRank(const Config &cfg, int rank, int nranks, int device,
                    int idFd, int outFd)
{
    g_logRank = rank;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = onTerm;
    // SA_RESTART: without it a signal landing inside a driver ioctl or an NCCL
    // bootstrap socket call surfaces as a spurious CUDA/NCCL error instead of
    // the coordinated stop that was intended.
    sa.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    RankMsg msg;

    // --- unique ID rendezvous -------------------------------------------------
    // Rank 0 mints the ID and hands it up to the supervisor, which fans it back
    // out to every other rank. The NCCL bootstrap root lives in rank 0's
    // process, which stays alive for the whole run, so this is safe. The
    // supervisor itself never touches CUDA or NCCL — it forks before any
    // context exists, which is the only way fork-per-GPU is legal at all.
    ncclUniqueId ncclId;
    memset(&ncclId, 0, sizeof(ncclId));
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&ncclId));
        if (!writeAll(outFd, &ncclId, sizeof(ncclId)))
            RANK_DIE("failed to publish ncclUniqueId to supervisor");
    } else {
        if (!readAll(idFd, &ncclId, sizeof(ncclId)))
            RANK_DIE("failed to receive ncclUniqueId from supervisor");
    }

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    cudaStream_t stream = nullptr; // legacy NULL stream, as DCGM/gpu-burn use
    if (cfg.explicitStream)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cublasHandle_t cub = nullptr;
    CUBLAS_CHECK(cublasCreate(&cub));
    CUBLAS_CHECK(cublasSetStream(cub, stream));
    if (cfg.alwaysTensor)
        CUBLAS_CHECK(cublasSetMathMode(cub, PCIEBURN_TENSOR_MATH));

    // --- NCCL buffers and comm come FIRST ------------------------------------
    // gpu-burn sizes its C array from 90% of free VRAM in one allocation, which
    // leaves nothing for NCCL. Reserving the collective buffers and building the
    // comm before sizing the GEMM working set means NCCL's internal channel
    // buffers are already accounted for by cudaMemGetInfo, with no guessing at
    // the overhead.
    DeviceBuffers db;
    CUDA_CHECK(cudaMalloc(&db.collSend, cfg.collMax));
    CUDA_CHECK(cudaMalloc(&db.collRecv, cfg.collMax));
    CUDA_CHECK(cudaMemset(db.collSend, 1, cfg.collMax));
    CUDA_CHECK(cudaMemset(db.collRecv, 0, cfg.collMax));
    CUDA_CHECK(cudaMalloc(&db.stopDev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db.reduceDev, sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc((void **)&db.stageInt, 4 * sizeof(int),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&db.stageU64, sizeof(unsigned long long),
                             cudaHostAllocDefault));
    memset(db.stageInt, 0, 4 * sizeof(int));
    *db.stageU64 = 0;
    CUDA_CHECK(cudaEventCreate(&db.evGemmStart));
    CUDA_CHECK(cudaEventCreate(&db.evGemmEnd));

    NCCL_CHECK(ncclCommInitRank(&g_comm, nranks, ncclId, rank));
    g_commLive = true;
    ncclComm_t comm = g_comm;

    logf_("device %d (%s) comm ready, %d ranks", device, prop.name, nranks);

    // --- GEMM working set ----------------------------------------------------
    const size_t dim   = cfg.matrixDim;
    const size_t rs64  = sizeof(double) * dim * dim;
    const size_t rs32  = rs64 / 2;
    const size_t rs16  = rs64 / 4;

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const size_t useBytes = (size_t)((double)freeB * cfg.memFrac);

    const size_t abBytes = 2 * (rs64 + rs32 + rs16);
    if (useBytes <= abBytes + rs64)
        RANK_DIE("insufficient free VRAM: %.2f GiB free, need > %.2f GiB",
                 (double)freeB / 1073741824.0,
                 (double)(abBytes + rs64) / 1073741824.0);

    size_t baseIters = (useBytes - abBytes) / rs64;
    if (cfg.maxCBuffers && baseIters > cfg.maxCBuffers)
        baseIters = cfg.maxCBuffers;
    if (baseIters < 1)
        baseIters = 1;

    // Ranks MUST agree on the iteration count. It is derived from free VRAM,
    // which differs per GPU (a display attached to one card is enough), and a
    // disagreement means ranks issue different numbers of collectives and the
    // whole job deadlocks. Reduce to the global minimum before allocating.
    {
        const unsigned long long local = (unsigned long long)baseIters;
        *db.stageU64 = local;
        CUDA_CHECK(cudaMemcpyAsync(db.reduceDev, db.stageU64,
                                   sizeof(*db.stageU64),
                                   cudaMemcpyHostToDevice, stream));
        NCCL_CHECK(ncclAllReduce(db.reduceDev, db.reduceDev, 1, ncclUint64,
                                 ncclMin, comm, stream));
        CUDA_CHECK(cudaMemcpyAsync(db.stageU64, db.reduceDev,
                                   sizeof(*db.stageU64),
                                   cudaMemcpyDeviceToHost, stream));
        const char *why = nullptr;
        if (!pollStream(stream, comm, cfg.collTimeout, &why))
            RANK_DIE("iteration-count agreement failed: %s", why);
        const unsigned long long agreed = *db.stageU64;
        if (agreed < 1)
            RANK_DIE("agreed iteration count is zero");
        if (agreed != local)
            logf_("iteration count reduced %llu -> %llu to match slowest rank",
                  local, agreed);
        baseIters = (size_t)agreed;
    }

    CUDA_CHECK(cudaMalloc(&db.aFP64, rs64));
    CUDA_CHECK(cudaMalloc(&db.bFP64, rs64));
    CUDA_CHECK(cudaMalloc(&db.aFP32, rs32));
    CUDA_CHECK(cudaMalloc(&db.bFP32, rs32));
    CUDA_CHECK(cudaMalloc(&db.aFP16, rs16));
    CUDA_CHECK(cudaMalloc(&db.bFP16, rs16));
    CUDA_CHECK(cudaMalloc(&db.faultyDev, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db.nanDev, sizeof(int)));

    db.base.resize(baseIters);
    for (size_t i = 0; i < baseIters; i++)
        CUDA_CHECK(cudaMalloc(&db.base[i], rs64));

    // FP32 aliases 2 matrices per base buffer, FP16 aliases 4 (DCGM layout).
    db.iters[PREC_DOUBLE] = baseIters;
    db.iters[PREC_SINGLE] = baseIters * 2;
    db.iters[PREC_HALF]   = baseIters * 4;
    for (size_t i = 0; i < baseIters; i++) {
        char *b = (char *)db.base[i];
        db.cHost[PREC_DOUBLE].push_back(b);
        for (int j = 0; j < 2; j++)
            db.cHost[PREC_SINGLE].push_back(b + (size_t)j * rs32);
        for (int j = 0; j < 4; j++)
            db.cHost[PREC_HALF].push_back(b + (size_t)j * rs16);
    }
    for (int p = 0; p < 3; p++) {
        const size_t n = db.cHost[p].size();
        CUDA_CHECK(cudaMalloc(&db.cDev[p], n * sizeof(void *)));
        CUDA_CHECK(cudaMemcpy(db.cDev[p], db.cHost[p].data(), n * sizeof(void *),
                              cudaMemcpyHostToDevice));
    }

    // Host A/B fill: srand(10) with values in [0,10), FP32/FP16 as downcasts of
    // the FP64 draw, A and B interleaved. Bit-identical to DCGM's AllocBuffers.
    {
        const size_t n = dim * dim;
        std::vector<double> hA64(n), hB64(n);
        std::vector<float>  hA32(n), hB32(n);
        std::vector<__half> hA16(n), hB16(n);
        srand(10);
        for (size_t i = 0; i < n; i++) {
            hA64[i] = ((double)(rand() % 1000000) / 100000.0);
            hA32[i] = (float)hA64[i];
            hA16[i] = __float2half(hA32[i]);
            hB64[i] = ((double)(rand() % 1000000) / 100000.0);
            hB32[i] = (float)hB64[i];
            hB16[i] = __float2half(hB32[i]);
        }
        CUDA_CHECK(cudaMemcpy(db.aFP64, hA64.data(), rs64, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.bFP64, hB64.data(), rs64, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.aFP32, hA32.data(), rs32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.bFP32, hB32.data(), rs32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.aFP16, hA16.data(), rs16, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.bFP16, hB16.data(), rs16, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    logf_("%zu base C buffers (%.2f GiB), %.2f GiB VRAM still free, "
          "iters half/single/double = %zu/%zu/%zu",
          baseIters, (double)(baseIters * rs64) / 1073741824.0,
          (double)freeB / 1073741824.0, db.iters[PREC_HALF],
          db.iters[PREC_SINGLE], db.iters[PREC_DOUBLE]);

    msgInit(&msg, rank, MSG_READY);
    // cudaDeviceProp::name is 256 bytes; bound it explicitly so the whole line
    // fits msg.text rather than relying on snprintf to clip it.
    snprintf(msg.text, sizeof(msg.text), "%.60s dev%d %zu bufs", prop.name,
             device, baseIters);
    writeAll(outFd, &msg, sizeof(msg));

    // --- main loop -----------------------------------------------------------
    const double startTime  = nowSec();
    const double perGemm    = gflopsPerGemm(cfg.matrixDim);
    long long outerIter     = 0;
    long long cumGemms = 0, cumColls = 0, cumFaulty = 0, cumNans = 0;
    long long cumCollBytes  = 0;
    double lastReport       = 0.0;
    bool hintedTensorCores  = cfg.alwaysTensor;
    bool stopping           = false;

    while (!stopping) {
        for (size_t pi = 0; pi < cfg.precisions.size() && !stopping; pi++) {
            const Precision prec = cfg.precisions[pi];
            const size_t iters   = db.iters[prec];
            const size_t collSz  = collSizeFor(cfg, outerIter);
            const size_t chunk   = cfg.gemmsPerColl ? cfg.gemmsPerColl : iters;

            const double passStart = nowSec();
            CUDA_CHECK(cudaEventRecord(db.evGemmStart, stream));

            // The interleave: a chunk of unpaced GEMMs, then a collective, on
            // the same stream so the collective is genuinely ordered behind the
            // compute — the transformer-layer shape, not merely concurrent load.
            size_t collsThisPass = 0;
            for (size_t from = 0; from < iters; from += chunk) {
                const size_t to = (from + chunk < iters) ? from + chunk : iters;
                gemmBurst(cfg, db, cub, prec, from, to);
                // Close the GEMM timing window after the last chunk's GEMMs but
                // before its collective, so throughput excludes collective time.
                if (to >= iters)
                    CUDA_CHECK(cudaEventRecord(db.evGemmEnd, stream));
                issueCollective(cfg, db, comm, stream, rank, nranks, collSz);
                collsThisPass++;
            }

            if (cfg.doCompare) {
                runCompare(cfg, db, prec, stream);
                CUDA_CHECK(cudaMemcpyAsync(&db.stageInt[0], db.faultyDev,
                                           sizeof(int), cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaMemcpyAsync(&db.stageInt[1], db.nanDev,
                                           sizeof(int), cudaMemcpyDeviceToHost,
                                           stream));
            } else {
                db.stageInt[0] = 0;
                db.stageInt[1] = 0;
            }

            // Coordinated stop. Every rank contributes its own wish to stop and
            // takes the max, so all ranks leave the loop on the SAME iteration.
            // Without this, SIGTERM landing at slightly different times would
            // leave ranks issuing mismatched collectives — a guaranteed hang on
            // the way out of an otherwise clean run.
            const double elapsed = nowSec() - startTime;
            db.stageInt[2] = (g_stopRequested || elapsed >= cfg.duration) ? 1 : 0;
            CUDA_CHECK(cudaMemcpyAsync(db.stopDev, &db.stageInt[2], sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            NCCL_CHECK(ncclAllReduce(db.stopDev, db.stopDev, 1, ncclInt, ncclMax,
                                     comm, stream));
            CUDA_CHECK(cudaMemcpyAsync(&db.stageInt[3], db.stopDev, sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));

            const char *why = nullptr;
            if (!pollStream(stream, comm, cfg.collTimeout, &why)) {
                msgInit(&msg, rank, MSG_ERROR);
                msg.outerIter = outerIter;
                msg.precision = (int32_t)prec;
                // msgInit just zeroed the struct, so carry the accumulated
                // counters explicitly. "How much traffic had moved when it
                // broke" is the datum this test exists to capture, and without
                // this the failing rank reports zeros for exactly the run that
                // matters.
                msg.gemms     = cumGemms;
                msg.colls     = cumColls;
                msg.collBytes = cumCollBytes;
                msg.faulty    = cumFaulty;
                msg.nans      = cumNans;
                snprintf(msg.text, sizeof(msg.text), "%s", why);
                writeAll(outFd, &msg, sizeof(msg));
                logf_("FAILURE at iter %lld (%s) after %lld GEMMs, %lld "
                      "collectives, %.2f GiB moved: %s",
                      outerIter, precName(prec), cumGemms, cumColls,
                      (double)cumCollBytes / 1073741824.0, why);
                rankAbort();
                _exit(EXIT_RANK_HANG);
            }

            // Safe to read the pinned staging block now that the stream drained.
            const int hFaulty    = db.stageInt[0];
            const int hNans      = db.stageInt[1];
            const int agreedStop = db.stageInt[3];

            cumGemms     += (long long)iters;
            cumColls     += (long long)collsThisPass;
            cumCollBytes += (long long)collsThisPass * (long long)collSz;
            cumFaulty    += hFaulty;
            cumNans      += hNans;

            // GPU time for the GEMM window only. Wall time for the whole pass
            // would fold in the collectives and the compare kernel, which is
            // what would make this number incomparable to a dcgmi diag baseline.
            // In chunked mode (--gemms-per-coll > 0) the window unavoidably
            // spans the interleaved collectives too.
            float gemmMs = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gemmMs, db.evGemmStart,
                                            db.evGemmEnd));
            const double gflops =
                gemmMs > 0.0f
                    ? (double)iters * perGemm / ((double)gemmMs / 1000.0)
                    : 0.0;
            const double passSec = nowSec() - passStart;

            // DCGM pulls in tensor cores at the halfway mark rather than from
            // the start; mirror that so the thermal/power ramp matches a diag run.
            if (!hintedTensorCores && elapsed > cfg.duration / 2.0) {
                hintedTensorCores = true;
                CUBLAS_CHECK(cublasSetMathMode(cub, PCIEBURN_TENSOR_MATH));
                logf_("enabled tensor-op math mode at %.1fs", elapsed);
            }

            if (agreedStop)
                stopping = true;

            const double nowT = nowSec();
            if (stopping || nowT - lastReport >= cfg.reportInterval) {
                lastReport = nowT;
                msgInit(&msg, rank, MSG_PROGRESS);
                msg.precision = (int32_t)prec;
                msg.outerIter = outerIter;
                msg.gemms     = cumGemms;
                msg.colls     = cumColls;
                msg.collBytes = cumCollBytes;
                msg.faulty    = cumFaulty;
                msg.nans      = cumNans;
                msg.gflops    = gflops;
                snprintf(msg.text, sizeof(msg.text), "%s coll=%zuMiB pass=%.2fs",
                         precName(prec), collSz >> 20, passSec);
                writeAll(outFd, &msg, sizeof(msg));
            }

            outerIter++;
        }
    }

    logf_("stopping cleanly after %lld iters, %lld GEMMs, %lld collectives "
          "(%.2f GiB moved), faulty=%lld nan=%lld",
          outerIter, cumGemms, cumColls,
          (double)cumCollBytes / 1073741824.0, cumFaulty, cumNans);

    msgInit(&msg, rank, MSG_DONE);
    msg.outerIter = outerIter;
    msg.gemms     = cumGemms;
    msg.colls     = cumColls;
    msg.collBytes = cumCollBytes;
    msg.faulty    = cumFaulty;
    msg.nans      = cumNans;
    writeAll(outFd, &msg, sizeof(msg));

    // Ordered teardown. All ranks agreed to stop on the same iteration, so
    // ncclCommDestroy cannot block here.
    g_commLive = false;
    ncclCommDestroy(comm);
    g_comm = nullptr;

    for (size_t i = 0; i < db.base.size(); i++)
        cudaFree(db.base[i]);
    for (int p = 0; p < 3; p++)
        cudaFree(db.cDev[p]);
    cudaFree(db.aFP64); cudaFree(db.bFP64);
    cudaFree(db.aFP32); cudaFree(db.bFP32);
    cudaFree(db.aFP16); cudaFree(db.bFP16);
    cudaFree(db.faultyDev); cudaFree(db.nanDev);
    cudaFree(db.collSend); cudaFree(db.collRecv);
    cudaFree(db.stopDev); cudaFree(db.reduceDev);
    cudaFreeHost(db.stageInt); cudaFreeHost(db.stageU64);
    cudaEventDestroy(db.evGemmStart); cudaEventDestroy(db.evGemmEnd);
    cublasDestroy(cub);
    if (stream)
        cudaStreamDestroy(stream);

    _exit((cumFaulty || cumNans) ? EXIT_RANK_FAILURE : 0);
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

// Binary suffixes, matching nccl-tests parsesize() so -b/-e values transfer.
static bool parseSize(const char *s, size_t *out)
{
    char *end = nullptr;
    double v  = strtod(s, &end);
    if (end == s || v < 0)
        return false;
    double mult = 1.0;
    if (*end) {
        switch (*end) {
        case 'K': case 'k': mult = 1024.0; break;
        case 'M': case 'm': mult = 1048576.0; break;
        case 'G': case 'g': mult = 1073741824.0; break;
        default: return false;
        }
        if (end[1] != '\0' && strcmp(end + 1, "B") && strcmp(end + 1, "iB"))
            return false;
    }
    *out = (size_t)(v * mult);
    return true;
}

static void usage(const char *argv0)
{
    printf(
"pcieburn — interleaved compute + NCCL/PCIe stress test (design Option B)\n"
"\n"
"Usage: %s [options]\n"
"\n"
"  --duration SEC        run length (default 60; the fault has reproduced\n"
"                        in 55-65s under dcgmi diag alone)\n"
"  --matrix-dim N        GEMM dimension (default 2048, DCGM's default)\n"
"  --precision LIST      comma list of half,single,double (default half,single,\n"
"                        matching DCGM's default set on a consumer GPU)\n"
"  --gemms-per-coll N    GEMMs between collectives; 0 = one full DCGM-style\n"
"                        burst per collective (default 0). Small values (8-64)\n"
"                        are closer to a real transformer layer.\n"
"  --collective NAME     allreduce | alltoall | sendrecv (default allreduce)\n"
"  --coll-min SIZE       smallest collective, binary suffixes (default 128M)\n"
"  --coll-max SIZE       largest collective (default 1G). Both send and recv\n"
"                        buffers are this size, so it costs 2x VRAM.\n"
"  --coll-factor N       sweep multiplier (default 2)\n"
"  --mem-frac F          fraction of remaining VRAM for C matrices (default 0.9)\n"
"  --max-c-buffers N     cap the C buffer count (default 0 = unlimited)\n"
"  --gpus LIST           comma list of device indices (default all visible)\n"
"  --stream MODE         legacy | explicit (default legacy, as DCGM/gpu-burn)\n"
"  --always-tensor       enable tensor-op math from the start, not at halfway\n"
"  --no-compare          skip result verification (pure load)\n"
"  --coll-timeout SEC    per-rank hang timeout (default 120, 0 = off)\n"
"  --watchdog SEC        supervisor silence timeout (default 60, 0 = off)\n"
"  --report-interval SEC progress cadence (default 1.0)\n"
"  --event-log PATH      append a CSV event log for telemetry correlation\n"
"  --tag NAME            label recorded in the event log\n"
"  -h, --help            this message\n"
"\n"
"Exit status: 0 all ranks clean; 1 usage/setup error; 2 a rank reported\n"
"faults or NaNs; 3 a rank hung or died (the signature this test hunts).\n", argv0);
}

static bool parseArgs(int argc, char **argv, Config &cfg)
{
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        auto next = [&](const char **out) -> bool {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", a);
                return false;
            }
            *out = argv[++i];
            return true;
        };
        const char *v = nullptr;

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            exit(0);
        } else if (!strcmp(a, "--duration")) {
            if (!next(&v)) return false;
            cfg.duration = atof(v);
        } else if (!strcmp(a, "--matrix-dim")) {
            if (!next(&v)) return false;
            cfg.matrixDim = (unsigned)atoi(v);
        } else if (!strcmp(a, "--precision")) {
            if (!next(&v)) return false;
            cfg.precisions.clear();
            std::string s(v);
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t c = s.find(',', pos);
                if (c == std::string::npos)
                    c = s.size();
                std::string t = s.substr(pos, c - pos);
                if (t == "half")        cfg.precisions.push_back(PREC_HALF);
                else if (t == "single") cfg.precisions.push_back(PREC_SINGLE);
                else if (t == "double") cfg.precisions.push_back(PREC_DOUBLE);
                else if (!t.empty()) {
                    fprintf(stderr, "unknown precision '%s'\n", t.c_str());
                    return false;
                }
                pos = c + 1;
            }
            if (cfg.precisions.empty()) {
                fprintf(stderr, "--precision needs at least one of half,single,double\n");
                return false;
            }
        } else if (!strcmp(a, "--gemms-per-coll")) {
            if (!next(&v)) return false;
            cfg.gemmsPerColl = (size_t)strtoull(v, nullptr, 10);
        } else if (!strcmp(a, "--collective")) {
            if (!next(&v)) return false;
            if (!strcmp(v, "allreduce"))      cfg.collective = COLL_ALLREDUCE;
            else if (!strcmp(v, "alltoall"))  cfg.collective = COLL_ALLTOALL;
            else if (!strcmp(v, "sendrecv"))  cfg.collective = COLL_SENDRECV;
            else {
                fprintf(stderr, "unknown collective '%s'\n", v);
                return false;
            }
        } else if (!strcmp(a, "--coll-min")) {
            if (!next(&v) || !parseSize(v, &cfg.collMin)) {
                fprintf(stderr, "bad --coll-min\n");
                return false;
            }
        } else if (!strcmp(a, "--coll-max")) {
            if (!next(&v) || !parseSize(v, &cfg.collMax)) {
                fprintf(stderr, "bad --coll-max\n");
                return false;
            }
        } else if (!strcmp(a, "--coll-factor")) {
            if (!next(&v)) return false;
            cfg.collFactor = (unsigned)atoi(v);
        } else if (!strcmp(a, "--mem-frac")) {
            if (!next(&v)) return false;
            cfg.memFrac = atof(v);
        } else if (!strcmp(a, "--max-c-buffers")) {
            if (!next(&v)) return false;
            cfg.maxCBuffers = (size_t)strtoull(v, nullptr, 10);
        } else if (!strcmp(a, "--gpus")) {
            if (!next(&v)) return false;
            cfg.gpus.clear();
            std::string s(v);
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t c = s.find(',', pos);
                if (c == std::string::npos)
                    c = s.size();
                std::string t = s.substr(pos, c - pos);
                if (!t.empty())
                    cfg.gpus.push_back(atoi(t.c_str()));
                pos = c + 1;
            }
        } else if (!strcmp(a, "--stream")) {
            if (!next(&v)) return false;
            if (!strcmp(v, "legacy"))        cfg.explicitStream = false;
            else if (!strcmp(v, "explicit")) cfg.explicitStream = true;
            else {
                fprintf(stderr, "--stream must be legacy or explicit\n");
                return false;
            }
        } else if (!strcmp(a, "--always-tensor")) {
            cfg.alwaysTensor = true;
        } else if (!strcmp(a, "--no-compare")) {
            cfg.doCompare = false;
        } else if (!strcmp(a, "--coll-timeout")) {
            if (!next(&v)) return false;
            cfg.collTimeout = atof(v);
        } else if (!strcmp(a, "--watchdog")) {
            if (!next(&v)) return false;
            cfg.watchdog = atof(v);
        } else if (!strcmp(a, "--report-interval")) {
            if (!next(&v)) return false;
            cfg.reportInterval = atof(v);
        } else if (!strcmp(a, "--event-log")) {
            if (!next(&v)) return false;
            cfg.eventLog = v;
        } else if (!strcmp(a, "--tag")) {
            if (!next(&v)) return false;
            cfg.tag = v;
        } else {
            fprintf(stderr, "unknown argument '%s' (try --help)\n", a);
            return false;
        }
    }

    if (cfg.precisions.empty()) {
        // DCGM's default on a GPU whose FP64:FP32 ratio excludes double, which
        // is every consumer part including the RTX 5090.
        cfg.precisions.push_back(PREC_HALF);
        cfg.precisions.push_back(PREC_SINGLE);
    }
    // A zero or sub-element collective size makes the sweep generator loop
    // forever: with the default factor of 2, size*2 never grows past 0.
    if (cfg.collMin < 1024) {
        fprintf(stderr, "--coll-min must be at least 1K (got %zu bytes)\n",
                cfg.collMin);
        return false;
    }
    if (cfg.collMax < cfg.collMin)
        cfg.collMax = cfg.collMin;
    if (cfg.matrixDim < 64 || cfg.matrixDim > 65536) {
        fprintf(stderr, "--matrix-dim %u out of range\n", cfg.matrixDim);
        return false;
    }
    // The FP32/FP16 views alias into each FP64-sized C buffer at offsets of
    // 4*dim^2 and 2*dim^2 bytes. A dim that is not a multiple of 8 leaves those
    // offsets under-aligned for cuBLAS's tensor-core kernels, giving either
    // NOT_SUPPORTED or silently wrong results — and wrong results would surface
    // as bogus faulty counts, indistinguishable from the fault being hunted.
    if (cfg.matrixDim % 8 != 0) {
        fprintf(stderr, "--matrix-dim must be a multiple of 8 (got %u)\n",
                cfg.matrixDim);
        return false;
    }
    if (cfg.memFrac <= 0.0 || cfg.memFrac >= 1.0) {
        fprintf(stderr, "--mem-frac must be in (0,1)\n");
        return false;
    }
    if (cfg.duration <= 0.0) {
        fprintf(stderr, "--duration must be positive\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Supervisor
// ---------------------------------------------------------------------------

// Device count is probed in a short-lived child. The supervisor must not
// initialize CUDA before forking the ranks — a CUDA context does not survive
// fork, so any context created here would poison every rank.
static int probeDeviceCount()
{
    int fds[2];
    if (pipe(fds) != 0)
        return -1;

    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) {
        close(fds[0]);
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess)
            n = -1;
        writeAll(fds[1], &n, sizeof(n));
        close(fds[1]);
        _exit(0);
    }

    close(fds[1]);
    int n = -1;
    if (!readAll(fds[0], &n, sizeof(n)))
        n = -1;
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return n;
}

struct RankState {
    pid_t pid       = -1;
    int   readFd    = -1;
    int   idFd      = -1;   // supervisor -> rank, for the unique ID
    bool  alive     = true;
    bool  ready     = false;
    bool  done      = false;
    bool  failed    = false;
    double lastSeen = 0.0;
    RankMsg last {};   // zero-init: the final summary reads this even for a
                       // rank that died before its first report
    // Peak GEMM throughput per precision. Tracked separately because half and
    // single differ by roughly 4x, so a single combined peak would report only
    // the half figure and hide the FP32 number entirely.
    double peakGflops[3] = {0.0, 0.0, 0.0};
};

static FILE *g_eventLog = nullptr;

static void eventLogWrite(const Config &cfg, const char *event, int rank,
                          const RankMsg *m, const char *note)
{
    if (!g_eventLog)
        return;
    char ts[48];
    isoNow(ts, sizeof(ts));

    // coll_bytes is the nominal buffer total, which understates real bus work.
    // Derive the two figures that actually matter for a PCIe investigation:
    // algorithmic bytes leaving the rank in one direction, and total bytes over
    // that rank's own link once host staging doubles every remote byte.
    const double nominal = m ? (double)m->collBytes : 0.0;
    const double algo = nominal * collBwFactor(cfg.collective,
                                               (int)cfg.gpus.size());
    const double link = algo * PCIE_HOST_STAGING_MULT;

    fprintf(g_eventLog,
            "%s,%s,%s,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.2f,%s\n",
            ts, cfg.tag.c_str(), event, rank,
            m ? (long long)m->outerIter : 0LL,
            m ? (long long)m->gemms : 0LL,
            m ? (long long)m->colls : 0LL,
            m ? (long long)m->collBytes : 0LL,
            (long long)algo,
            (long long)link,
            m ? (long long)m->faulty : 0LL,
            m ? (long long)m->nans : 0LL,
            m ? m->gflops : 0.0,
            note ? note : "");
    fflush(g_eventLog);
}

int main(int argc, char **argv)
{
    Config cfg;
    if (!parseArgs(argc, argv, cfg))
        return 1;

    setvbuf(stdout, nullptr, _IOLBF, 0);

    // If a rank dies during startup its pipe closes, and the supervisor's write
    // of the unique ID would raise SIGPIPE and kill the supervisor at default
    // disposition — leaving every other rank alive and orphaned holding a CUDA
    // context. writeAll already reports EPIPE, so ignoring the signal routes
    // that case into the existing error handling.
    signal(SIGPIPE, SIG_IGN);

    // Ctrl-C delivers SIGINT to the whole foreground process group. Ranks handle
    // it and begin a coordinated stop, but without a handler here the supervisor
    // would die immediately and never reach the reap loop, orphaning them.
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = onTerm;
        sa.sa_flags   = SA_RESTART;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
    }

    if (cfg.gpus.empty()) {
        int n = probeDeviceCount();
        if (n <= 0) {
            fprintf(stderr,
                    "no CUDA devices found (probe returned %d). Is the driver "
                    "loaded?\n", n);
            return 1;
        }
        for (int i = 0; i < n; i++)
            cfg.gpus.push_back(i);
    }

    const int nranks = (int)cfg.gpus.size();
    if (nranks < 2) {
        fprintf(stderr,
                "pcieburn needs at least 2 GPUs — the whole point is inter-GPU "
                "collectives over PCIe. Got %d.\n", nranks);
        return 1;
    }

    if (!cfg.eventLog.empty()) {
        const bool fresh = access(cfg.eventLog.c_str(), F_OK) != 0;
        g_eventLog = fopen(cfg.eventLog.c_str(), "a");
        if (!g_eventLog) {
            fprintf(stderr, "cannot open event log %s: %s\n",
                    cfg.eventLog.c_str(), strerror(errno));
            return 1;
        }
        if (fresh)
            fprintf(g_eventLog,
                    "timestamp,tag,event,rank,outer_iter,gemms,colls,"
                    "coll_bytes_nominal,coll_bytes_algorithmic,"
                    "coll_bytes_pcie_link,faulty,nans,gflops,note\n");
    }

    {
        std::string gpuList;
        for (int i = 0; i < nranks; i++) {
            char b[16];
            snprintf(b, sizeof(b), "%s%d", i ? "," : "", cfg.gpus[i]);
            gpuList += b;
        }
        std::string precList;
        for (size_t i = 0; i < cfg.precisions.size(); i++) {
            precList += (i ? "," : "");
            precList += precName(cfg.precisions[i]);
        }
        logf_("pcieburn starting: %d ranks on GPUs [%s]", nranks, gpuList.c_str());
        logf_("  duration=%.0fs matrix_dim=%u precision=%s stream=%s",
              cfg.duration, cfg.matrixDim, precList.c_str(),
              cfg.explicitStream ? "explicit" : "legacy");
        const std::string burstDesc =
            cfg.gemmsPerColl ? std::to_string(cfg.gemmsPerColl) : "full-burst";
        logf_("  collective=%s sweep %zuMiB..%zuMiB x%u (%zu sizes) "
              "gemms_per_coll=%s",
              collName(cfg.collective), cfg.collMin >> 20, cfg.collMax >> 20,
              cfg.collFactor, collSizeCount(cfg), burstDesc.c_str());
        eventLogWrite(cfg, "start", -1, nullptr, "supervisor up");
    }

    std::vector<RankState> ranks(nranks);

    // Pipes first, then fork everything. Two pipes per rank: rank -> supervisor
    // for status, and supervisor -> rank for the unique ID broadcast.
    // A setup failure part-way through the fork loop must take down the ranks
    // already started, or returning would leave them running with live CUDA
    // contexts and no supervisor.
    auto abortStartup = [&](int upTo, const char *what) {
        fprintf(stderr, "%s: %s\n", what, strerror(errno));
        for (int q = 0; q < upTo; q++) {
            if (ranks[q].pid > 0) {
                kill(ranks[q].pid, SIGKILL);
                waitpid(ranks[q].pid, nullptr, 0);
            }
        }
    };

    for (int r = 0; r < nranks; r++) {
        int up[2], down[2];
        if (pipe(up) != 0) {
            abortStartup(r, "pipe() failed");
            return 1;
        }
        if (pipe(down) != 0) {
            close(up[0]);
            close(up[1]);
            abortStartup(r, "pipe() failed");
            return 1;
        }

        pid_t pid = fork();
        if (pid < 0) {
            close(up[0]);
            close(up[1]);
            close(down[0]);
            close(down[1]);
            abortStartup(r, "fork() failed");
            return 1;
        }
        if (pid == 0) {
            // Child: close every fd belonging to other ranks so a dying rank
            // actually produces EOF on its pipe instead of being held open.
            close(up[0]);
            close(down[1]);
            for (int q = 0; q < r; q++) {
                if (ranks[q].readFd >= 0) close(ranks[q].readFd);
                if (ranks[q].idFd >= 0)   close(ranks[q].idFd);
            }
            if (g_eventLog) {
                fclose(g_eventLog);
                g_eventLog = nullptr; // ranks must never write to a closed FILE*
            }
            runRank(cfg, r, nranks, cfg.gpus[r], down[0], up[1]);
            _exit(0); // unreachable
        }

        close(up[1]);
        close(down[0]);
        ranks[r].pid    = pid;
        ranks[r].readFd = up[0];
        ranks[r].idFd   = down[1];
        ranks[r].lastSeen = nowSec();
    }

    // Rank 0 publishes the unique ID; fan it out to the rest.
    {
        ncclUniqueId id;
        if (!readAll(ranks[0].readFd, &id, sizeof(id))) {
            logf_("rank 0 failed to produce an ncclUniqueId; aborting");
            for (int r = 0; r < nranks; r++)
                kill(ranks[r].pid, SIGKILL);
            for (int r = 0; r < nranks; r++)
                waitpid(ranks[r].pid, nullptr, 0);
            return 3;
        }
        for (int r = 1; r < nranks; r++) {
            if (!writeAll(ranks[r].idFd, &id, sizeof(id)))
                logf_("failed to hand unique ID to rank %d", r);
        }
    }
    // Ranks 1..N-1 have the ID; closing the write ends prevents a stuck reader.
    for (int r = 0; r < nranks; r++) {
        if (ranks[r].idFd >= 0) {
            close(ranks[r].idFd);
            ranks[r].idFd = -1;
        }
    }

    // Startup (NCCL init plus ~90% of VRAM in per-buffer allocations on every
    // rank) is not covered by the watchdog, so it gets its own deadline.
    const double STARTUP_DEADLINE_SEC = 300.0;

    const double startTime = nowSec();
    double loadStart       = startTime; // reset once every rank reports ready
    bool   timersReset     = false;
    int    readyCount      = 0;
    double maxGap          = 0.0;      // largest observed inter-report gap
    bool sawFault    = false;
    bool sawHang     = false;
    bool killed      = false;
    bool interrupted = false;
    int  liveRanks   = nranks;

    // A rank that stops participating hangs every other rank, so the supervisor
    // does not wait politely: the first loss takes the whole group down. That is
    // also the detection path for the fault this test exists to provoke.
    double killTime = 0.0;
    auto killAll = [&](const char *why) {
        if (killed)
            return;
        killed   = true;
        killTime = nowSec();
        logf_("tearing down all ranks: %s", why);
        eventLogWrite(cfg, "killall", -1, nullptr, why);
        for (int r = 0; r < nranks; r++)
            if (ranks[r].alive)
                kill(ranks[r].pid, SIGTERM);
    };

    while (liveRanks > 0) {
        if (g_stopRequested && !killed) {
            interrupted = true;
            logf_("signal received — stopping ranks");
            killAll("interrupted by signal");
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        int maxFd = -1;
        for (int r = 0; r < nranks; r++) {
            if (ranks[r].readFd >= 0) {
                FD_SET(ranks[r].readFd, &rfds);
                if (ranks[r].readFd > maxFd)
                    maxFd = ranks[r].readFd;
            }
        }
        if (maxFd < 0)
            break;

        struct timeval tv;
        tv.tv_sec  = 0;
        tv.tv_usec = 250000;
        int sel = select(maxFd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel < 0) {
            if (errno == EINTR)
                continue;
            logf_("select() failed: %s", strerror(errno));
            break;
        }

        const double now = nowSec();

        for (int r = 0; r < nranks && sel > 0; r++) {
            if (ranks[r].readFd < 0 || !FD_ISSET(ranks[r].readFd, &rfds))
                continue;
            sel--;

            RankMsg m;
            if (!readAll(ranks[r].readFd, &m, sizeof(m))) {
                // EOF: the rank exited. Whether that was clean is decided by
                // its exit status below.
                close(ranks[r].readFd);
                ranks[r].readFd = -1;
                ranks[r].alive  = false;
                liveRanks--;
                if (!ranks[r].done) {
                    logf_("rank %d pipe closed without reporting done", r);
                    eventLogWrite(cfg, "rank_lost", r, &ranks[r].last,
                                  "pipe closed unexpectedly");
                    sawHang = true;
                    killAll("a rank exited unexpectedly");
                }
                continue;
            }

            if (m.magic != MSG_MAGIC) {
                // Writes are a fixed 176 bytes, well under PIPE_BUF, so this
                // should be unreachable. Treat it as a lost rank rather than
                // continuing — ignoring it would spin at select speed forever.
                logf_("rank %d sent a malformed message; treating as lost", r);
                eventLogWrite(cfg, "bad_message", r, &ranks[r].last,
                              "protocol desync");
                close(ranks[r].readFd);
                ranks[r].readFd = -1;
                ranks[r].alive  = false;
                liveRanks--;
                sawHang = true;
                killAll("rank protocol desync");
                continue;
            }

            // Track the largest gap between reports; the teardown grace period
            // is derived from it, since a rank can only notice a stop request
            // once per precision pass.
            if (ranks[r].ready) {
                const double gap = now - ranks[r].lastSeen;
                if (gap > maxGap)
                    maxGap = gap;
            }
            ranks[r].lastSeen = now;
            ranks[r].last     = m;

            switch (m.kind) {
            case MSG_READY:
                ranks[r].ready = true;
                readyCount++;
                logf_("rank %d ready (%s)", r, m.text);
                eventLogWrite(cfg, "ready", r, &m, m.text);
                // Ranks time their own duration from after their allocations, so
                // the supervisor's deadlines must start from the same point.
                // Anchoring them at fork time instead would let a slow startup
                // trip the watchdog or the overrun backstop on a healthy run —
                // a false "hang" verdict, the one result that would poison this
                // investigation.
                if (readyCount == nranks && !timersReset) {
                    timersReset = true;
                    loadStart   = nowSec();
                    for (int q = 0; q < nranks; q++)
                        ranks[q].lastSeen = loadStart;
                    logf_("all %d ranks ready after %.1fs startup; load phase "
                          "begins now", nranks, loadStart - startTime);
                    eventLogWrite(cfg, "all_ready", -1, nullptr,
                                  "load phase start");
                }
                break;

            case MSG_PROGRESS:
                logf_("rank %d iter=%lld %s gemms=%lld colls=%lld "
                      "moved=%.1fGiB %.0f GFLOP/s faulty=%lld nan=%lld",
                      r, (long long)m.outerIter, m.text, (long long)m.gemms,
                      (long long)m.colls,
                      (double)m.collBytes / 1073741824.0, m.gflops,
                      (long long)m.faulty, (long long)m.nans);
                eventLogWrite(cfg, "progress", r, &m, m.text);
                if (m.precision >= 0 && m.precision < 3 &&
                    m.gflops > ranks[r].peakGflops[m.precision])
                    ranks[r].peakGflops[m.precision] = m.gflops;
                if (m.faulty || m.nans) {
                    if (!sawFault)
                        logf_("rank %d reported COMPUTE FAULTS "
                              "(faulty=%lld nan=%lld)", r,
                              (long long)m.faulty, (long long)m.nans);
                    sawFault = true;
                }
                break;

            case MSG_DONE:
                ranks[r].done = true;
                logf_("rank %d done: %lld iters, %lld GEMMs, %lld collectives, "
                      "%.2f GiB moved, faulty=%lld nan=%lld", r,
                      (long long)m.outerIter, (long long)m.gemms,
                      (long long)m.colls,
                      (double)m.collBytes / 1073741824.0,
                      (long long)m.faulty, (long long)m.nans);
                eventLogWrite(cfg, "done", r, &m, m.text);
                if (m.faulty || m.nans)
                    sawFault = true;
                break;

            case MSG_ERROR:
                ranks[r].failed = true;
                logf_("rank %d ERROR at iter %lld: %s", r,
                      (long long)m.outerIter, m.text);
                eventLogWrite(cfg, "error", r, &m, m.text);
                sawHang = true;
                killAll("a rank reported a failure");
                break;

            default:
                break;
            }
        }

        // Watchdog. A GPU that has fallen off the bus produces exactly this:
        // silence. Catching it here is what puts a timestamp on the event.
        if (cfg.watchdog > 0.0 && !killed && timersReset) {
            for (int r = 0; r < nranks; r++) {
                if (!ranks[r].alive || ranks[r].done || !ranks[r].ready)
                    continue;
                const double silent = now - ranks[r].lastSeen;
                if (silent > cfg.watchdog) {
                    logf_("rank %d silent for %.1fs (watchdog %.1fs) — "
                          "SUSPECTED HANG", r, silent, cfg.watchdog);
                    eventLogWrite(cfg, "watchdog", r, &ranks[r].last,
                                  "rank silent past watchdog");
                    sawHang = true;
                    killAll("watchdog timeout");
                    break;
                }
            }
        }

        // Startup deadline. The watchdog is gated on ranks being ready, so a
        // hang before the first report needs its own bound.
        if (!timersReset && !killed &&
            nowSec() - startTime > STARTUP_DEADLINE_SEC) {
            logf_("only %d/%d ranks became ready within %.0fs — aborting",
                  readyCount, nranks, STARTUP_DEADLINE_SEC);
            eventLogWrite(cfg, "startup_timeout", -1, nullptr,
                          "not all ranks ready");
            sawHang = true;
            killAll("startup timeout");
        }

        // Hard backstop: ranks stop themselves via the coordinated stop flag,
        // so overrunning the duration by a wide margin means something is stuck.
        // Measured from loadStart, not fork time, so a slow startup cannot trip it.
        if (!killed && timersReset &&
            nowSec() - loadStart > cfg.duration + 60.0 + cfg.reportInterval) {
            logf_("ranks overran duration by >60s — forcing shutdown");
            eventLogWrite(cfg, "overrun", -1, nullptr, "duration exceeded");
            sawHang = true;
            killAll("duration overrun");
        }

        // A rank wedged in the driver never closes its pipe, so liveRanks would
        // never reach zero and this loop would spin forever — which is exactly
        // the case a hung GPU produces. Leave the loop after a grace period and
        // let the reap stage escalate to SIGKILL.
        //
        // The grace period must exceed one precision pass, because that is how
        // often a rank can notice the stop request. A fixed 15s would mean any
        // run with a long pass always escalates to SIGKILL, so the clean
        // shutdown path would never actually be exercised.
        const double grace =
            (maxGap > 0.0 && 3.0 * maxGap > 20.0) ? 3.0 * maxGap : 20.0;
        if (killed && nowSec() - killTime > grace) {
            logf_("ranks did not exit within %.0fs of teardown; escalating",
                  grace);
            break;
        }
    }

    // Reap, escalating to SIGKILL for anything that ignored SIGTERM. A GPU that
    // is truly wedged can leave a process unkillable in the kernel; say so
    // rather than hanging here forever.
    for (int r = 0; r < nranks; r++) {
        int status = 0;
        for (int attempt = 0; attempt < 100; attempt++) {
            pid_t got = waitpid(ranks[r].pid, &status, WNOHANG);
            if (got == ranks[r].pid)
                goto reaped;
            if (got < 0) {
                // waitpid failed (ECHILD and friends). Falling through to decode
                // the untouched status would read WIFEXITED(0)/WEXITSTATUS(0)
                // and silently record a lost rank as a clean pass.
                logf_("waitpid(rank %d, pid %d) failed: %s", r,
                      (int)ranks[r].pid, strerror(errno));
                eventLogWrite(cfg, "waitpid_failed", r, &ranks[r].last,
                              strerror(errno));
                sawHang = true;
                goto nextRank;
            }
            if (attempt == 20)
                kill(ranks[r].pid, SIGKILL);
            usleep(100000);
        }
        logf_("rank %d (pid %d) will not die — likely stuck in the driver. "
              "A node power cycle may be required.", r, (int)ranks[r].pid);
        eventLogWrite(cfg, "unkillable", r, &ranks[r].last,
                      "process did not exit after SIGKILL");
        sawHang = true;
        continue;

    reaped:
        if (WIFEXITED(status)) {
            const int code = WEXITSTATUS(status);
            if (code == EXIT_RANK_HANG) {
                sawHang = true;
            } else if (code == EXIT_RANK_FAILURE) {
                sawFault = true;
            } else if (code != 0) {
                logf_("rank %d exited with status %d", r, code);
                sawHang = true;
            }
        } else if (WIFSIGNALED(status)) {
            const int sig = WTERMSIG(status);
            // SIGTERM is how a coordinated teardown ends, so it is only
            // notable if we did not ask for it.
            if (!killed || (sig != SIGTERM && sig != SIGKILL)) {
                logf_("rank %d killed by signal %d", r, sig);
                sawHang = true;
            }
        }
    nextRank:;
    }

    const double wall     = nowSec() - startTime;
    const double loadWall = timersReset ? nowSec() - loadStart : 0.0;

    long long totalGemms = 0, totalColls = 0, totalBytes = 0;
    long long totalFaulty = 0, totalNans = 0;
    for (int r = 0; r < nranks; r++) {
        totalGemms  += ranks[r].last.gemms;
        totalColls  += ranks[r].last.colls;
        totalBytes  += ranks[r].last.collBytes;
        totalFaulty += ranks[r].last.faulty;
        totalNans   += ranks[r].last.nans;
    }

    logf_("=== run complete in %.1fs (%.1fs startup, %.1fs under load) ===",
          wall, loadStart - startTime, loadWall);
    logf_("  ranks=%d GEMMs=%lld collectives=%lld faulty=%lld nan=%lld",
          nranks, totalGemms, totalColls, totalFaulty, totalNans);

    // Bus accounting. The raw counter is nominal buffer bytes, which understates
    // real PCIe work: a ring collective moves more than its buffer, and with P2P
    // disabled every remote byte crosses the bus twice via host RAM.
    {
        const double GiB      = 1073741824.0;
        const double bwFactor = collBwFactor(cfg.collective, nranks);
        const double perRankNominal =
            nranks ? (double)totalBytes / (double)nranks : 0.0;
        const double perRankAlgo = perRankNominal * bwFactor;
        const double perRankLink = perRankAlgo * PCIE_HOST_STAGING_MULT;
        const double egressGBps =
            loadWall > 0.0 ? perRankAlgo / 1e9 / loadWall : 0.0;
        const double linkGBps =
            loadWall > 0.0 ? perRankLink / 1e9 / loadWall : 0.0;

        logf_("  PCIe traffic (%s, %d ranks, bw factor %.2fx, host-staged %.0fx):",
              collName(cfg.collective), nranks, bwFactor,
              PCIE_HOST_STAGING_MULT);
        logf_("    nominal buffers            %9.2f GiB/rank  %10.2f GiB total",
              perRankNominal / GiB, (double)totalBytes / GiB);
        logf_("    algorithmic (one way)      %9.2f GiB/rank  %10.2f GiB total",
              perRankAlgo / GiB, perRankAlgo * nranks / GiB);
        logf_("    PCIe link (both ways)      %9.2f GiB/rank  %10.2f GiB total",
              perRankLink / GiB, perRankLink * nranks / GiB);
        logf_("    achieved                   %9.2f GB/s egress/rank "
              "(%.1f%% of Gen5 x16), %.2f GB/s link/rank",
              egressGBps, 100.0 * egressGBps / PCIE_GEN5_X16_GBPS, linkGBps);
    }

    // Per-GPU peak GEMM throughput. A rank consistently below its peers here is
    // a signal in its own right, independent of any PCIe fault.
    logf_("  per-GPU peak GFLOP/s (GEMM window only):");
    for (int r = 0; r < nranks; r++) {
        char buf[256];
        int off = snprintf(buf, sizeof(buf), "    rank %d (dev %d):", r,
                           cfg.gpus[r]);
        for (size_t p = 0; p < cfg.precisions.size(); p++) {
            if (off < 0 || off >= (int)sizeof(buf))
                break;
            const Precision pr = cfg.precisions[p];
            const int w = snprintf(buf + off, sizeof(buf) - (size_t)off,
                                   "  %s %.0f", precName(pr),
                                   ranks[r].peakGflops[pr]);
            if (w < 0)
                break;
            off += w;

            // Machine-readable copy, one row per rank per precision.
            RankMsg pk {};
            pk.magic     = MSG_MAGIC;
            pk.rank      = r;
            pk.precision = (int32_t)pr;
            pk.gflops    = ranks[r].peakGflops[pr];
            eventLogWrite(cfg, "peak", r, &pk, precName(pr));
        }
        logf_("%s", buf);
    }

    int rc = 0;
    const char *verdict;
    if (sawHang) {
        verdict = "RANK LOST OR HUNG — check dmesg for Xid, and rasdaemon for AER";
        rc = 3;
    } else if (sawFault) {
        verdict = "COMPUTE FAULTS detected";
        rc = 2;
    } else if (interrupted) {
        verdict = "interrupted by signal before the duration elapsed";
        rc = 0;
    } else {
        verdict = "clean";
        rc = 0;
    }
    logf_("  verdict: %s", verdict);
    eventLogWrite(cfg, "finish", -1, nullptr, verdict);

    if (g_eventLog)
        fclose(g_eventLog);
    return rc;
}
