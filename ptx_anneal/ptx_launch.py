# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PTX-direct launch: assemble a FIXED PTX with ptxas (optionally applying an ACF) and run the
resulting cubin via the CUDA driver API -- no frontend recompile from source.

This is the core of the PTX-direct factory objective. Tuning here means:

    ptxas <fixed kernel.ptx> [--apply-controls=<acf>] -o cand.cubin   # the per-candidate assemble
    cuModuleLoadData(cand.cubin) -> cuLaunchKernel(...)               # the objective body (run + time)

Because the PTX is frozen, what we tune is byte-identical to production and the whole
"did source recompile to the same PTX" parity problem disappears.

Launch ABI: the kernel params are the NON-constexpr, NON-`equal_to_1`-specialized scalar/pointer
args in source order, followed by two trailing internal pointers -- `global_scratch` then
`profile_scratch` -- which are NULL when their sizes are 0. The PTX `.visible .entry` param list is
authoritative for the count/types; callers build `kernel_args` to match and we assert parity against
the PTX before launching.
"""

import ctypes
import math
import re
import subprocess
import tempfile

# cuda-python is needed only to LAUNCH (driver API). Import lazily so the pure helpers
# (parse_entry / build_spec / ptxas_compile) can be used from a dependency-light context
# (e.g. a frontend's collection path) without requiring cuda-python.
try:
    from cuda.bindings import driver as _drv
except Exception:  # noqa: BLE001 - pragma: no cover; optional dep, only the launch path needs it.
    # Deliberately blind: cuda-python can fail on import for reasons beyond ImportError (a driver
    # mismatch, a partially installed wheel). This module must stay importable regardless -- the
    # spec/parse/ptxas half of it is used from dependency-light contexts, and every test relies on
    # that.
    _drv = None


# --------------------------------------------------------------------------------------
# driver error handling
# --------------------------------------------------------------------------------------
def _chk(ret):
    """Unwrap a cuda.bindings (CUresult, *rest) tuple, raising on a non-success status."""
    err = ret[0]
    if err != _drv.CUresult.CUDA_SUCCESS:
        name = _drv.cuGetErrorString(err)[1]
        try:
            name = name.decode()
        except (AttributeError, UnicodeDecodeError):
            pass  # already str, or undecodable bytes -- either way, report it as-is below
        raise RuntimeError(f"CUDA driver error: {err} ({name})")
    rest = ret[1:]
    return rest[0] if len(rest) == 1 else rest


_INITED = False


def _ensure_init(device: int = 0):
    """cuInit once and make sure a CUDA context is current on this thread. We retain+bind the
    device PRIMARY context -- the same one torch uses -- so driver launches and torch tensors share
    a context (and so this works even before torch has run its first device op in a fresh process)."""
    global _INITED
    if _INITED:
        return
    if _drv is None:
        raise RuntimeError(
            "cuda-python (cuda.bindings.driver) is required to launch cubins; "
            "the launch path needs it even though build_spec/ptxas do not."
        )
    _chk(_drv.cuInit(0))
    cur = _drv.cuCtxGetCurrent()
    has_ctx = cur[0] == _drv.CUresult.CUDA_SUCCESS and int(getattr(cur[1], "_ptr", cur[1]) or 0) != 0
    if not has_ctx:
        dev = _chk(_drv.cuDeviceGet(device))
        ctx = _chk(_drv.cuDevicePrimaryCtxRetain(dev))
        _chk(_drv.cuCtxSetCurrent(ctx))
    _INITED = True


# --------------------------------------------------------------------------------------
# PTX parsing
# --------------------------------------------------------------------------------------
_ENTRY_RE = re.compile(r"\.visible\s+\.entry\s+([\w$]+)\s*\(([^)]*)\)", re.DOTALL)
_TARGET_RE = re.compile(r"\.target\s+(sm_\w+)")


def parse_arch(ptx: str) -> str:
    m = _TARGET_RE.search(ptx)
    if not m:
        raise ValueError("no .target sm_* found in PTX")
    return m.group(1)


def _param_kind(p: str) -> str:
    """Map a PTX `.param` declaration to a launch-arg kind: 'tma' | 'ptr' | 'i64' | 'f64' | 'f32' | 'i32'."""
    if ".b8" in p and "[" in p:
        return "tma"  # a by-value byte-array param == a 128B CUtensorMap (TMA descriptor)
    if ".ptr" in p:
        return "ptr"
    if ".u64" in p or ".s64" in p or ".b64" in p:
        return "i64"
    if ".f64" in p:
        return "f64"
    if ".f32" in p:
        return "f32"
    return "i32"  # .u32 / .s32 / .b32 / narrower ints (passed in a 32-bit slot)


def parse_entry(ptx: str):
    """Return (entry_name, [kind, ...]) for the single .visible .entry, kind per _param_kind."""
    m = _ENTRY_RE.search(ptx)
    if not m:
        raise ValueError("no .visible .entry found in PTX")
    name = m.group(1)
    params = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return name, [_param_kind(p) for p in params]


def _specialize_ordered_args(ordered_args):
    """Apply the frontend's param specialization to the bound-order args: drop constexprs and
    equal_to_1 integer scalars (bool -> int, kept), and return the surviving
    ("tensor" | "tensordesc" | "scalar", ...) entries in order -- the ones that map onto real kernel
    params. Shared by build_spec (PTX-driven) and build_spec_from_schema (schema-driven) so the two
    apply identical specialization."""
    survivors = []
    for a in ordered_args:
        if a[0] == "constexpr":
            continue
        if a[0] in ("tensor", "tensordesc"):
            survivors.append(a)
        elif a[0] == "scalar":
            v = a[1]
            if isinstance(v, bool):
                survivors.append(("scalar", int(v)))
            elif isinstance(v, int) and v == 1:
                continue
            else:
                survivors.append(("scalar", v))
        else:
            raise NotImplementedError(f"unsupported arg kind {a[0]!r}")
    return survivors


def _is_pointer_type(ty: str) -> bool:
    """True if a Triton signature type string denotes a global pointer (e.g. '*fp16', '*i8')."""
    return ty.startswith("*")


def _scalar_kind_from_type(ty: str) -> str:
    """Map a Triton scalar signature type ('i32', 'i64', 'fp32', 'fp64', ...) to a launch-arg kind,
    matching what _param_kind derives from the PTX param for the same argument."""
    t = ty.lstrip("*")
    if t in ("i64", "u64"):
        return "i64"
    if t in ("fp64", "f64"):
        return "f64"
    if t in ("fp32", "f32"):
        return "f32"
    if t and t[0] in ("i", "u"):  # i1/i8/i16/i32/u32/... ride a 32-bit slot
        return "i32"
    raise NotImplementedError(f"unsupported scalar schema type {ty!r}")


def build_spec(ptx: str, metadata, grid, ordered_args, ptxas: str) -> dict:
    """Build the source-free launch spec from a compiled kernel + its (bound-order) call args.

    ordered_args: list in source/bound order, each one of:
        ("tensor", {"id", "shape", "dtype", "strides"})
        ("tensordesc", {"base": {"id","shape","dtype","strides"}, "desc_shape", "desc_strides",
                        "block_shape", "padding"})   # host-side TMA TensorDescriptor
        ("scalar", value) | ("constexpr",)
    Applies the frontend's specialization (drop constexprs and `equal_to_1` integer scalars), maps the
    survivors onto the PTX `.entry` param slots -- including expanding each TensorDescriptor into its
    [tensormap, *shape, *strides] params (the TMA ABI) -- and appends the two trailing null scratch
    pointers. Backing tensors are deduped by `id` so descriptors that share a tensor share one
    allocation.

    Cluster shape is read from ``metadata.ctas_per_cga`` ONLY: that is the sole field TLX kernels set to
    request a cluster (the CUDA-native path), and the nvidia backend mirrors it into ``cluster_dims`` in
    ``Metadata.__post_init__`` -- so ctas_per_cga is authoritative and cluster_dims/cluster are not read.

    Raises NotImplementedError for kernels this PTX-direct path does not cover yet (non-null
    global/profile scratch, multi-CTA/cluster, fp4-padded/im2col TMA, other arg types, or any param
    mismatch) so the caller can fail open and simply not collect that kernel."""
    name, kinds = parse_entry(ptx)
    gss = int(getattr(metadata, "global_scratch_size", 0) or 0)
    pss = int(getattr(metadata, "profile_scratch_size", 0) or 0)
    num_ctas = int(getattr(metadata, "num_ctas", 1) or 1)
    # Cluster shape from ctas_per_cga only (see docstring): the field TLX kernels populate; the backend
    # mirrors it into cluster_dims, so ctas_per_cga is authoritative.
    cluster = list(getattr(metadata, "ctas_per_cga", None) or [1, 1, 1])
    tdmeta = list(getattr(metadata, "tensordesc_meta", None) or [])
    if gss or pss:
        raise NotImplementedError(f"non-null scratch (global={gss} profile={pss})")
    # Cluster support is limited to the TLX / CUDA-native path: num_ctas == 1 with the cluster shape
    # carried by the cubin's `.reqnctapercluster` directive, so a plain launch with the total-CTA grid
    # re-forms the clusters (matches the frontend launcher, which sets NO CLUSTER_DIMENSION attr for
    # this path). The Triton num_ctas>1 path (grid multiplied by num_ctas + an explicit
    # CLUSTER_DIMENSION launch attr) is not covered.
    if num_ctas != 1:
        raise NotImplementedError(f"multi-CTA via num_ctas>1 (num_ctas={num_ctas})")
    cluster = [int(c) for c in cluster]
    cluster_size = 1
    for c in cluster:
        cluster_size *= c
    if len(kinds) < 2 or kinds[-2] != "ptr" or kinds[-1] != "ptr":
        raise NotImplementedError("expected two trailing scratch pointer params")
    user_kinds = kinds[:-2]  # drop global_scratch + profile_scratch

    tensors = []
    _id2idx = {}

    def _tensor_idx(info):
        key = info.get("id")
        if key is not None and key in _id2idx:
            return _id2idx[key]
        idx = len(tensors)
        tensors.append({"shape": list(info["shape"]), "dtype": info["dtype"], "strides": list(info["strides"])})
        if key is not None:
            _id2idx[key] = idx
        return idx

    # Specialize: drop constexprs and equal_to_1 int scalars (what the frontend removes from params).
    survivors = _specialize_ordered_args(ordered_args)

    args, tensordescs, ki, tdmi = [], [], 0, 0

    def _kind_at(i):
        if i >= len(user_kinds):
            raise NotImplementedError("param parity: ran past PTX user params")
        return user_kinds[i]

    for s in survivors:
        if s[0] == "tensor":
            if _kind_at(ki) != "ptr":
                raise NotImplementedError("tensor arg mapped to a non-pointer PTX param")
            args.append({"t": _tensor_idx(s[1])})
            ki += 1
        elif s[0] == "scalar":
            k = _kind_at(ki)
            if k in ("ptr", "tma"):
                raise NotImplementedError("scalar arg mapped to a pointer/tma PTX param")
            args.append({k: int(s[1]) if k in ("i32", "i64") else float(s[1])})
            ki += 1
        else:  # tensordesc -> [tensormap, *desc_shape, *desc_strides]
            info = s[1]
            if tdmi >= len(tdmeta):
                raise NotImplementedError("more tensordesc args than tensordesc_meta entries")
            meta = tdmeta[tdmi]
            tdmi += 1
            if meta.get("fp4_padded") or meta.get("is_im2col"):
                raise NotImplementedError("fp4-padded / im2col TMA not supported yet")
            if _kind_at(ki) != "tma":
                raise NotImplementedError("tensordesc arg not aligned to a tensormap PTX param")
            rank = len(info["desc_shape"])
            args.append({"tma": len(tensordescs)})
            tensordescs.append(
                {
                    "base": _tensor_idx(info["base"]),
                    "desc_shape": list(info["desc_shape"]),
                    "desc_strides": list(info["desc_strides"]),
                    "swizzle": int(meta["swizzle"]),
                    "elem_size": int(meta["elem_size"]),
                    "elem_type": int(meta["elem_type"]),
                    "block_size": list(meta["block_size"]),
                    "padding": int(info.get("padding", 0)),
                }
            )
            ki += 1
            for d in range(rank):  # *shape params
                k = _kind_at(ki)
                args.append({k: int(info["desc_shape"][d])})
                ki += 1
            for d in range(rank):  # *stride params
                k = _kind_at(ki)
                args.append({k: int(info["desc_strides"][d])})
                ki += 1

    if ki != len(user_kinds):
        raise NotImplementedError(f"param parity: consumed {ki} != {len(user_kinds)} PTX user params")
    args += [{"null": True}, {"null": True}]  # global_scratch, profile_scratch

    g = [int(grid[0]), int(grid[1]) if len(grid) > 1 else 1, int(grid[2]) if len(grid) > 2 else 1]
    # A clustered launch re-forms clusters from the total-CTA grid via `.reqnctapercluster`; the driver
    # requires each grid dim to be a multiple of its cluster dim (as it was in the collected run).
    if cluster_size != 1 and any(gd % cd != 0 for gd, cd in zip(g, cluster, strict=True)):
        raise NotImplementedError(f"grid {g} not divisible by cluster {cluster}")
    spec = {
        "entry": name,
        "arch": parse_arch(ptx),
        "shared": int(getattr(metadata, "shared", 0) or 0),
        "block": [int(getattr(metadata, "num_warps", 1) or 1) * 32, 1, 1],
        "grid": g,
        "ptxas": ptxas,
        "tensors": tensors,
        "args": args,
    }
    if cluster_size != 1:
        spec["cluster"] = cluster
    if tensordescs:
        spec["tensordescs"] = tensordescs
    return spec


def build_spec_from_schema(schema: dict, ptx: str, grid, ordered_args, ptxas: str) -> dict:
    """PROTOTYPE alternative to build_spec: source the post-specialization param list from the
    compiler's Level 0 launch-metadata schema (``kernel.asm["launch_metadata"]``, see the nvidia
    backend's make_launch_metadata) instead of parsing the PTX .entry and re-deriving specialization.
    Produces the IDENTICAL spec.json shape as build_spec, so the factory replay path is unchanged.

    Why: build_spec re-implements the frontend's specialization + PTX-param ABI, so the kernel arg
    list lives in two places and can silently drift. The schema is the compiler's own source of
    truth -- it carries the entry name, num_warps/num_ctas, shared, cluster_dims, scratch sizes, and
    the args[] list (Triton types, constants already dropped). This maps captured runtime VALUES
    (tensor allocations, scalar values from ordered_args) onto schema["args"] 1:1, so no PTX param
    parsing is needed.

    Scope (prototype): tensor + scalar params only. tensordesc/TMA args raise NotImplementedError --
    the Level 0 schema lists a descriptor as ONE arg and does not carry the flattened
    [tensormap, *shape, *strides] leaf kinds (that flattening lives in expand_signature / the Level 1
    C launcher), so full TMA coverage needs PTX or Level 1, not Level 0 alone. ``arch`` is still read
    from the PTX .target, which the schema does not carry."""
    num_ctas = int(schema.get("num_ctas") or 1)
    cluster = [int(c) for c in (schema.get("cluster_dims") or [1, 1, 1])]
    gss = int(schema.get("global_scratch_size") or 0)
    pss = int(schema.get("profile_scratch_size") or 0)
    num_warps = int(schema.get("num_warps") or 1)
    shared = int(schema.get("shared_mem") or 0)
    schema_args = schema.get("args") or []

    if gss or pss:
        raise NotImplementedError(f"non-null scratch (global={gss} profile={pss})")
    if num_ctas != 1:
        raise NotImplementedError(f"multi-CTA via num_ctas>1 (num_ctas={num_ctas})")
    cluster_size = 1
    for c in cluster:
        cluster_size *= c

    survivors = _specialize_ordered_args(ordered_args)
    if len(survivors) != len(schema_args):
        raise NotImplementedError(
            f"schema/args parity: {len(schema_args)} schema params != {len(survivors)} runtime args"
        )

    tensors = []
    _id2idx = {}

    def _tensor_idx(info):
        key = info.get("id")
        if key is not None and key in _id2idx:
            return _id2idx[key]
        idx = len(tensors)
        tensors.append({"shape": list(info["shape"]), "dtype": info["dtype"], "strides": list(info["strides"])})
        if key is not None:
            _id2idx[key] = idx
        return idx

    args = []
    for sarg, s in zip(schema_args, survivors, strict=True):
        ty = str(sarg.get("type", ""))
        if s[0] == "tensordesc":
            raise NotImplementedError("tensordesc/TMA: Level 0 schema lacks flattened leaf kinds (needs PTX/Level 1)")
        if s[0] == "tensor":
            if not _is_pointer_type(ty):
                raise NotImplementedError(f"tensor arg mapped to non-pointer schema type {ty!r}")
            args.append({"t": _tensor_idx(s[1])})
        else:  # scalar
            k = _scalar_kind_from_type(ty)
            args.append({k: int(s[1]) if k in ("i32", "i64") else float(s[1])})
    args += [{"null": True}, {"null": True}]  # global_scratch, profile_scratch

    g = [int(grid[0]), int(grid[1]) if len(grid) > 1 else 1, int(grid[2]) if len(grid) > 2 else 1]
    if cluster_size != 1 and any(gd % cd != 0 for gd, cd in zip(g, cluster, strict=True)):
        raise NotImplementedError(f"grid {g} not divisible by cluster {cluster}")
    spec = {
        "entry": schema.get("entry_name") or "",
        "arch": parse_arch(ptx),
        "shared": shared,
        "block": [num_warps * 32, 1, 1],
        "grid": g,
        "ptxas": ptxas,
        "tensors": tensors,
        "args": args,
    }
    if cluster_size != 1:
        spec["cluster"] = cluster
    return spec


# --------------------------------------------------------------------------------------
# ptxas
# --------------------------------------------------------------------------------------
def ptxas_compile(ptx: str, ptxas: str, arch: str | None = None, acf_path: str | None = None, extra_args=()) -> bytes:
    """Assemble PTX -> cubin with ptxas. If acf_path is given, append --apply-controls=<acf>
    (the consume mechanism). Raises RuntimeError with ptxas stderr on failure."""
    arch = arch or parse_arch(ptx)
    with (
        tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=True) as pf,
        tempfile.NamedTemporaryFile(suffix=".cubin", delete=True) as cf,
    ):
        pf.write(ptx)
        pf.flush()
        cmd = [ptxas, f"-arch={arch}", pf.name, "-o", cf.name, *extra_args]
        if acf_path:
            cmd.append(f"--apply-controls={acf_path}")
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"ptxas failed (rc={r.returncode}): {r.stderr.strip()}\n  cmd: {' '.join(cmd)}")
        with open(cf.name, "rb") as f:
            return f.read()


# --------------------------------------------------------------------------------------
# driver load + launch
# --------------------------------------------------------------------------------------
def _build_kernel_params(kernel_args):
    """Build the ``void** kernelParams`` array (+ the ctypes value holders that back it) for
    ``cuLaunchKernel`` from the ordered launch-arg list. Returns ``(arr, holders)``; keep BOTH alive
    for the whole time the array is used (a plain launch, or a captured graph's lifetime).

    kernel_args: ('ptr', addr) | ('i32'|'i64', int) | ('f32', float) | ('null',) | ('tma', addr).
    For 'tma' the 128B tensormap is passed BY VALUE, so that slot points straight at the descriptor
    bytes and the caller must keep the backing tensormap objects alive."""
    arr = (ctypes.c_void_p * len(kernel_args))()
    holders = []  # keep ctypes value objects alive as long as `arr` is used
    for i, a in enumerate(kernel_args):
        kind = a[0]
        if kind == "tma":  # by-value tensormap: slot points directly at the 128B descriptor
            arr[i] = ctypes.c_void_p(int(a[1]))
            continue
        if kind == "ptr":
            h = ctypes.c_void_p(int(a[1]))
        elif kind == "null":
            h = ctypes.c_void_p(0)
        elif kind == "i32":
            h = ctypes.c_int32(int(a[1]))
        elif kind == "i64":
            h = ctypes.c_int64(int(a[1]))
        elif kind == "f32":
            h = ctypes.c_float(float(a[1]))
        else:
            raise ValueError(f"unsupported kernel arg kind: {kind}")
        holders.append(h)
        arr[i] = ctypes.cast(ctypes.pointer(h), ctypes.c_void_p)
    return arr, holders


class _PreparedLaunch:
    """Internal: a pre-built launch (fixed grid/block dims + the kernelParams array, held alive).

    Building the params ONCE gives a stable ``kernelParams`` pointer that can be re-issued cheaply and,
    critically, captured into a CUDA graph -- a graph records the params pointer and reuses it on every
    replay, so it must outlive the graph. Constructed inside ``LoadedKernel.launch`` / ``bench_ms_cudagraph``;
    callers never build one directly."""

    def __init__(self, grid, block, arr, holders):
        self.gx, self.gy, self.gz = (tuple(grid) + (1, 1, 1))[:3]
        self.bx, self.by, self.bz = (tuple(block) + (1, 1, 1))[:3]
        self.arr = arr
        self.holders = holders


class LoadedKernel:
    """A cubin loaded into the current context, with optional CTA cluster launch dimensions."""

    def __init__(self, cubin: bytes, name: str, shared: int, *, cluster=None):
        self.cluster = tuple(cluster) if cluster is not None else (1, 1, 1)
        if len(self.cluster) != 3 or any(not isinstance(c, int) or c < 1 for c in self.cluster):
            raise ValueError(f"invalid cluster dimensions: {self.cluster}")
        _ensure_init()
        self.module = _chk(_drv.cuModuleLoadData(cubin))
        self.func = _chk(_drv.cuModuleGetFunction(self.module, name.encode()))
        self.shared = int(shared)
        if self.shared > 48 * 1024:  # opt-in to >48KB dynamic smem
            _chk(
                _drv.cuFuncSetAttribute(
                    self.func,
                    _drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    self.shared,
                )
            )

    def unload(self):
        if getattr(self, "module", None) is not None:
            _drv.cuModuleUnload(self.module)
            self.module = None

    @classmethod
    def from_cubin(cls, cubin: bytes, name: str, shared: int) -> "LoadedKernel":
        return cls(cubin, name, shared)

    def _prepare(self, grid, block, kernel_args) -> "_PreparedLaunch":
        """Internal: build the kernelParams array once (stable pointer for reuse / graph capture)."""
        dims = (tuple(grid) + (1, 1, 1))[:3]
        if any(g % c for g, c in zip(dims, self.cluster, strict=True)):
            raise ValueError(f"grid {dims} is not divisible by cluster {self.cluster}")
        arr, holders = _build_kernel_params(kernel_args)
        return _PreparedLaunch(grid, block, arr, holders)

    def _launch_prepared(self, prepared: "_PreparedLaunch", stream=0):
        """Internal: launch a pre-built :class:`_PreparedLaunch` on ``stream`` (0 = default; a real
        stream is used for graph capture)."""
        p = prepared
        if self.cluster != (1, 1, 1):
            attr = _drv.CUlaunchAttribute()
            attr.id = _drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.cluster
            config = _drv.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = p.gx, p.gy, p.gz
            config.blockDimX, config.blockDimY, config.blockDimZ = p.bx, p.by, p.bz
            config.sharedMemBytes = self.shared
            config.hStream = stream
            config.attrs = [attr]
            config.numAttrs = 1
            _chk(_drv.cuLaunchKernelEx(config, self.func, int(ctypes.addressof(p.arr)), 0))
            return
        _chk(
            _drv.cuLaunchKernel(
                self.func,
                p.gx,
                p.gy,
                p.gz,
                p.bx,
                p.by,
                p.bz,
                self.shared,
                stream,
                int(ctypes.addressof(p.arr)),
                0,
            )
        )

    def launch(self, grid, block, kernel_args, stream=0):
        """kernel_args: ordered list of ('ptr', addr) | ('i32', int) | ('i64', int) | ('f32', float) |
        ('null',) | ('tma', addr_of_128B_tensormap). Builds the void** kernelParams array the frontend's
        launcher would build and calls cuLaunchKernel (cuLaunchKernelEx for clusters).
        For 'tma' the param IS passed by value (128
        bytes), so the kernelParams slot points straight at the tensormap bytes -- the caller must keep
        the backing PyCUtensorMap objects alive across the launch."""
        self._launch_prepared(self._prepare(grid, block, kernel_args), stream)


def load_cubin(cubin: bytes, name: str, shared: int, *, cluster=None) -> LoadedKernel:
    """Load cubin bytes into the current CUDA context and return a launchable handle."""
    return LoadedKernel(cubin, name, shared, cluster=cluster)


# --------------------------------------------------------------------------------------
# CUDA-graph benchmarking
#
# Times back-to-back GPU execution of ONE kernel with the per-launch HOST overhead (Python +
# cuLaunchKernel) removed from the timed window: capture N unrolled launches into a graph once, then
# time replays -- each cuGraphLaunch re-issues all N with a single host call. `measured = overhead +
# N*true`, so N is picked so N*true >> overhead (auto: estimate per-call, N = rep_ms / est). Reports
# the MIN per-launch ms over `reps` replays (robust to jitter). This removes only host jitter; GPU-side
# variance (DVFS/thermal/power/co-tenant) needs clock-locking + interleaved A/B at a higher layer.
# --------------------------------------------------------------------------------------
def _pick_n(est_ms: float, rep_ms: float, n_max: int = 100000) -> int:
    """Unroll count N so the captured replay region ~= rep_ms: N = clamp(rep_ms/est, 1, n_max)."""
    if not (est_ms > 0) or not math.isfinite(est_ms):
        return 1
    return max(1, min(n_max, int(rep_ms / est_ms)))


# Coarse pilot: how many event-timed launches to estimate per-launch ms with, only to size the graph's
# unroll count N. Not the measurement (that is N x reps graph replays), so it can be small; ~100 makes
# the estimate stable at negligible cost (a warmup already ran).
_ESTIMATE_LAUNCHES = 100


def _estimate_launch_ms(kernel: LoadedKernel, prepared: "_PreparedLaunch", stream) -> float:
    """Rough mean ms per launch (event-timed pilot) used ONLY to size the graph unroll count N.
    Coarse by design -- the trustworthy timing is the graph replay of N launches x reps."""
    start = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
    end = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
    try:
        _chk(_drv.cuEventRecord(start, stream))
        for _ in range(_ESTIMATE_LAUNCHES):
            kernel._launch_prepared(prepared, stream)
        _chk(_drv.cuEventRecord(end, stream))
        _chk(_drv.cuEventSynchronize(end))
        return float(_chk(_drv.cuEventElapsedTime(start, end))) / _ESTIMATE_LAUNCHES
    finally:
        _drv.cuEventDestroy(start)
        _drv.cuEventDestroy(end)


def bench_ms_cudagraph(
    kernel: LoadedKernel, grid, block, kernel_args, *, warmup: int = 25, rep_ms: float = 100.0, reps: int = 10
) -> float:
    """Per-launch ms via CUDA-graph replay (host launch overhead amortized out of the timed window).

    Give it the same launch inputs as ``LoadedKernel.launch`` (grid, block, kernel_args); the kernelParams
    are built once internally (stable pointer for capture) and reused across warmup/estimate/capture/replay.
    N (the number of launches stitched into the graph) is auto-sized so the replay region ~= rep_ms."""
    _ensure_init()
    prepared = kernel._prepare(grid, block, kernel_args)
    stream = _chk(_drv.cuStreamCreate(_drv.CUstream_flags.CU_STREAM_NON_BLOCKING))
    try:
        for _ in range(max(warmup, 1)):  # warm up clocks/caches on the capture stream
            kernel._launch_prepared(prepared, stream)
        _chk(_drv.cuStreamSynchronize(stream))

        n = _pick_n(_estimate_launch_ms(kernel, prepared, stream), rep_ms)

        # Capture N unrolled launches (THREAD_LOCAL mode; capture cannot use the default stream 0).
        _chk(_drv.cuStreamBeginCapture(stream, _drv.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
        for _ in range(n):
            kernel._launch_prepared(prepared, stream)
        graph = _chk(_drv.cuStreamEndCapture(stream))
        gexec = _chk(_drv.cuGraphInstantiate(graph, 0))
        try:
            best = math.inf
            for _ in range(max(reps, 1)):
                start = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
                end = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
                _chk(_drv.cuEventRecord(start, stream))
                _chk(_drv.cuGraphLaunch(gexec, stream))
                _chk(_drv.cuEventRecord(end, stream))
                _chk(_drv.cuEventSynchronize(end))
                best = min(best, float(_chk(_drv.cuEventElapsedTime(start, end))) / n)
                _drv.cuEventDestroy(start)
                _drv.cuEventDestroy(end)
            return best
        finally:
            _drv.cuGraphExecDestroy(gexec)
            _drv.cuGraphDestroy(graph)
    finally:
        _drv.cuStreamDestroy(stream)


def bench_ms_do_bench(
    kernel: LoadedKernel, grid, block, kernel_args, *, warmup: int = 25, reps: int = 50, flush_mb: int = 256
) -> float:
    """Per-launch ms as the MEDIAN of event-timed single launches, flushing the L2 before each rep.

    Mirrors ``triton.testing.do_bench`` (L2-cold, median) rather than the back-to-back graph replay of
    ``bench_ms_cudagraph`` (L2-hot, min): it is the metric faithful to the consumer's launch-time A/B, at
    the cost of DRAM-bound noise + more wall-clock. Same launch inputs as ``LoadedKernel.launch``."""
    _ensure_init()
    prepared = kernel._prepare(grid, block, kernel_args)
    stream = _chk(_drv.cuStreamCreate(_drv.CUstream_flags.CU_STREAM_NON_BLOCKING))
    flush_bytes = max(flush_mb, 1) * 1024 * 1024
    flush = _chk(_drv.cuMemAlloc(flush_bytes))
    try:
        for _ in range(max(warmup, 1)):  # warm up clocks/caches on the stream
            kernel._launch_prepared(prepared, stream)
        _chk(_drv.cuStreamSynchronize(stream))

        samples = []
        for _ in range(max(reps, 1)):
            _chk(_drv.cuMemsetD8Async(flush, 0, flush_bytes, stream))  # evict the L2 so each launch is cold
            start = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
            end = _chk(_drv.cuEventCreate(_drv.CUevent_flags.CU_EVENT_DEFAULT))
            _chk(_drv.cuEventRecord(start, stream))
            kernel._launch_prepared(prepared, stream)
            _chk(_drv.cuEventRecord(end, stream))
            _chk(_drv.cuEventSynchronize(end))
            samples.append(float(_chk(_drv.cuEventElapsedTime(start, end))))
            _drv.cuEventDestroy(start)
            _drv.cuEventDestroy(end)
        samples.sort()
        return samples[len(samples) // 2]  # median
    finally:
        _drv.cuMemFree(flush)
        _drv.cuStreamDestroy(stream)


# --------------------------------------------------------------------------------------
# Launch spec helpers -- turn a spec.json + allocated tensors into a launch.
#
# A spec (what build_spec emits, what a frontend writes as spec.json) is the self-contained launch
# ABI: entry/arch/shared/block/grid + the post-specialization arg order, so a run needs only
# kernel.ptx + spec.json -- no source. "args" is the FINAL kernel-param order (post equal_to_1 /
# constexpr specialization) including the two trailing null scratch params; {"t":i} is a pointer to
# tensors[i].
# --------------------------------------------------------------------------------------
def alloc_tensors(spec, device="cuda", seed=0):
    """Allocate the spec's tensors (deterministic, so baseline and ACF runs see identical inputs)."""
    import torch

    torch.manual_seed(seed)
    dts = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    out = []
    for t in spec["tensors"]:
        dt = dts[t["dtype"]]
        x = torch.empty_strided(t["shape"], t["strides"], dtype=dt, device=device)
        if dt.is_floating_point and dt.itemsize == 1:
            # FP8 has no normal_ kernel: sample float32 over the whole (possibly strided) storage
            # and let copy_ cast it down.
            flat = torch.empty(0, dtype=dt, device=device).set_(x.untyped_storage())
            flat.copy_(torch.randn(flat.numel(), device=device))
        elif dt.is_floating_point:
            x.normal_()
        else:
            x.zero_()
        out.append(x)
    return out


def build_tensormaps(spec, tensors):
    """Build the 128B CUtensorMap for each spec tensordesc via the frontend's tensormap encoder, and
    return (objs, addrs): the PyCUtensorMap objects (KEEP ALIVE across launch) and the address of each
    one's 128-byte tensormap (id(obj) + (basicsize - 128)). Empty for non-TMA kernels."""
    tds = spec.get("tensordescs")
    if not tds:
        return [], []
    import triton

    try:
        from triton.backends.nvidia.driver import TMA_DTYPE_DEVICE_TO_HOST as _D2H
    except Exception:  # noqa: BLE001 - private triton API; absent OR moved OR renamed across
        _D2H = {i: i for i in range(16)}  # versions, so fall back to identity rather than break.
    fill = triton.runtime.driver.active.utils.fill_tma_descriptor_tiled
    objs, addrs = [], []
    for td in tds:
        base = tensors[td["base"]]
        tm = fill(
            base.data_ptr(),
            td["swizzle"],
            td["elem_size"],
            _D2H[td["elem_type"]],
            list(td["block_size"]),
            list(td["desc_shape"]),
            list(td["desc_strides"]),
            td["padding"],
        )
        objs.append(tm)
        addrs.append(id(tm) + (type(tm).__basicsize__ - 128))
    return objs, addrs


def kernel_args_from_spec(spec, tensors, tensormap_addrs=None):
    """Turn the spec's final arg list + allocated tensors (+ tensormap addrs) into launch() args."""
    tensormap_addrs = tensormap_addrs or []
    ka = []
    for a in spec["args"]:
        if "t" in a:
            ka.append(("ptr", tensors[a["t"]].data_ptr()))
        elif "tma" in a:
            ka.append(("tma", tensormap_addrs[a["tma"]]))
        elif a.get("null"):
            ka.append(("null",))
        elif "i32" in a:
            ka.append(("i32", a["i32"]))
        elif "i64" in a:
            ka.append(("i64", a["i64"]))
        elif "f32" in a:
            ka.append(("f32", a["f32"]))
        else:
            raise ValueError(f"unsupported spec arg: {a}")
    return ka
