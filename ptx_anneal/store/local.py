# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""``LocalStore`` — a filesystem artifact store.

Layout (all files for one kernel share a directory)::

    <root>/<arch>/<hash16>/<hash16>.<toolchain>.acf        # version-tagged artifact
    <root>/<arch>/<hash16>/<hash16>.<toolchain>.acf.json   # provenance sidecar
    <root>/<arch>/<hash16>/<hash16>.<toolchain>.acf.cubin  # optional assembled-cubin cache
    <root>/<arch>/<hash16>/<hash16>.acf                  # no toolchain version given

``hash16`` is the first 16 characters of ``ir_hash``. The sidecar retains the full
hash, which is also returned by listings when the sidecar is available.

``target`` is recorded in the sidecar. Both the ``.acf`` and the assembled-cubin
cache are tagged by toolchain (ptxas) version: an ACF is opaque controls applied at
assemble time and is only valid for the ptxas that produced it (a mismatched ptxas can
reject it), so version-tagging lets one kernel keep a distinct tuned ACF per ptxas
version. The kernel *identity* (``ir_hash``) stays version-independent. When no
toolchain version is given, the untagged ``<hash16>.acf`` filename is used.

Reads and listings also support full-length hash directories and the older flat
``<root>/<arch>/<filename>`` layout, preferring short nested paths, then full nested paths.
New writes always use the short kernel directory.
Consumers constructing paths directly must adopt the nested layout to find new artifacts.

The store stays free of any frontend/engine dependency: it takes a precomputed
``ir_hash`` (the ``ptx`` target computes it) and opaque bytes, nothing more.
"""

from __future__ import annotations

import json
import os
import re

from .base import Store

# Interop env var (external contract — shared with the frontend; do NOT rename).
DEFAULT_STORE_ENV = "COMPILE_IQ_STORE"


def default_store_root() -> str:
    """Where artifacts live by default (overridable via ``COMPILE_IQ_STORE``)."""
    return os.environ.get(DEFAULT_STORE_ENV, os.path.expanduser("~/.compile_iq/store"))


def _toolchain_tag(version: str) -> str:
    """A filesystem-safe, bounded tag for a toolchain (ptxas) version string."""
    return re.sub(r"[^0-9A-Za-z._]", "", version or "")[:32] or "unknown"


def _acf_name(ir_hash: str, toolchain_version: str) -> str:
    return f"{ir_hash}.{_toolchain_tag(toolchain_version)}.acf" if toolchain_version else f"{ir_hash}.acf"


class LocalStore(Store):
    """A content-addressed artifact store backed by a local directory tree."""

    def __init__(self, root: str | None = None):
        self.root = root or default_store_root()

    # --- key -> path -----------------------------------------------------------------------------
    def _acf_path(self, arch: str, ir_hash: str, toolchain_version: str = "") -> str:
        short_hash = ir_hash[:16]
        return os.path.join(self.root, arch, short_hash, _acf_name(short_hash, toolchain_version))

    def _cubin_path(self, arch: str, ir_hash: str, toolchain_version: str) -> str:
        return self._acf_path(arch, ir_hash, toolchain_version or "unknown") + ".cubin"

    def _read_paths(self, arch: str, ir_hash: str, toolchain_version: str) -> list[str]:
        hashes = list(dict.fromkeys((ir_hash[:16], ir_hash)))
        nested = [os.path.join(self.root, arch, h, _acf_name(h, toolchain_version)) for h in hashes]
        flat = [os.path.join(self.root, arch, _acf_name(h, toolchain_version)) for h in hashes]
        return nested + flat

    def _read_path(self, arch: str, ir_hash: str, toolchain_version: str) -> str:
        paths = self._read_paths(arch, ir_hash, toolchain_version)
        return next((p for p in paths if os.path.exists(p)), paths[0])

    @staticmethod
    def _meta_at(acf_path: str) -> dict | None:
        try:
            with open(acf_path + ".json") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # --- Store ABC -------------------------------------------------------------------------------
    def has(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bool:
        return os.path.exists(self._read_path(arch, ir_hash, toolchain_version))

    def read(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bytes | None:
        p = self._read_path(arch, ir_hash, toolchain_version)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def read_meta(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> dict | None:
        # Choose the sidecar paired with the selected ACF, never metadata from the other layout.
        return self._meta_at(self._read_path(arch, ir_hash, toolchain_version))

    def write(
        self,
        target: str,
        arch: str,
        ir_hash: str,
        data: bytes,
        meta: dict | None = None,
        toolchain_version: str = "",
    ) -> str:
        p = self._acf_path(arch, ir_hash, toolchain_version)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"  # write+rename so a concurrent reader never sees a partial file
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        sidecar = {"target": target, "arch": arch, **(meta or {}), "ir_hash": ir_hash}
        with open(p + ".json", "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        return p

    def list(self, target: str | None = None, arch: str | None = None) -> list[dict]:
        out: list[dict] = []
        if not os.path.isdir(self.root):
            return out
        for arch_name in sorted(os.listdir(self.root)):
            if arch is not None and arch_name != arch:
                continue
            arch_dir = os.path.join(self.root, arch_name)
            if not os.path.isdir(arch_dir):
                continue
            paths: dict[tuple[str, str], tuple[tuple[bool, bool], str, str]] = {}
            for name in sorted(os.listdir(arch_dir)):
                path = os.path.join(arch_dir, name)
                nested = os.path.isdir(path)
                for fn in sorted(os.listdir(path)) if nested else [name]:
                    artifact = os.path.join(path, fn) if nested else path
                    if not fn.endswith(".acf") or not os.path.isfile(artifact):
                        continue
                    ir_hash, _, toolchain = fn.removesuffix(".acf").partition(".")
                    if nested and ir_hash != name:
                        continue
                    key = (ir_hash[:16], toolchain)
                    # Prefer short nested, full nested, short flat, then full flat artifacts.
                    rank = (not nested, len(ir_hash) > 16)
                    if key not in paths or rank < paths[key][0]:
                        paths[key] = (rank, artifact, ir_hash)
            for (_, toolchain), (_, path, ir_hash) in sorted(paths.items()):
                meta = self._meta_at(path) or {}
                if target is not None and meta.get("target", target) != target:
                    continue
                out.append(
                    {
                        "target": meta.get("target"),
                        "arch": arch_name,
                        "ir_hash": meta.get("ir_hash") or ir_hash,
                        "toolchain_version": toolchain or None,
                        "path": path,
                        "meta": meta,
                    }
                )
        return out

    # --- assembled-cubin cache (optional; not part of the Store ABC) ------------------------------
    def read_cubin(self, arch: str, ir_hash: str, toolchain_version: str) -> bytes | None:
        paths = self._read_paths(arch, ir_hash, toolchain_version or "unknown")
        acf = next((p for p in paths if os.path.exists(p)), None)
        # Pair the cubin with the selected ACF. With no ACF, support standalone cache entries.
        for path in [acf] if acf else paths:
            if os.path.exists(path + ".cubin"):
                with open(path + ".cubin", "rb") as f:
                    return f.read()
        return None

    def write_cubin(self, arch: str, ir_hash: str, toolchain_version: str, data: bytes) -> str:
        p = self._cubin_path(arch, ir_hash, toolchain_version)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        return p
