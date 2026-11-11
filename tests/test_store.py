# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the local artifact store (ptx_anneal.store.LocalStore)."""

import json
import os
from pathlib import Path

import pytest

from ptx_anneal.store import LocalStore, default_store_root

ARCH = "sm_100a"
IR_HASH = "6cb1f53d5136f5b8e3e3c25727534f6d006f6564d26b0e0b51cfd4826e254000"
HASH16 = IR_HASH[:16]
ACF = b"\x00\x01\x02fake-acf-bytes"


def test_write_read_roundtrip(tmp_path):
    s = LocalStore(str(tmp_path))
    assert not s.has("ptx", ARCH, IR_HASH)
    assert s.read("ptx", ARCH, IR_HASH) is None

    p = s.write("ptx", ARCH, IR_HASH, ACF, meta={"engine": "baseline", "best_ms": 1.23})
    assert s.has("ptx", ARCH, IR_HASH)
    assert s.read("ptx", ARCH, IR_HASH) == ACF

    meta = s.read_meta("ptx", ARCH, IR_HASH)
    assert meta is not None
    assert meta["target"] == "ptx" and meta["arch"] == ARCH and meta["ir_hash"] == IR_HASH
    assert meta["engine"] == "baseline"
    assert p == os.path.join(str(tmp_path), ARCH, HASH16, f"{HASH16}.acf")
    assert Path(p + ".json").parent == tmp_path / ARCH / HASH16
    assert s.read("ptx", ARCH, HASH16) == ACF
    assert s.read_meta("ptx", ARCH, HASH16)["ir_hash"] == IR_HASH


def test_list(tmp_path):
    s = LocalStore(str(tmp_path))
    assert s.list() == []
    s.write("ptx", ARCH, IR_HASH, ACF, meta={"engine": "baseline"})
    entries = s.list()
    assert len(entries) == 1
    e = entries[0]
    assert e["arch"] == ARCH and e["ir_hash"] == IR_HASH and e["target"] == "ptx"
    # filters
    assert s.list(arch="sm_90a") == []
    assert len(s.list(arch=ARCH)) == 1
    assert len(s.list(target="ptx")) == 1
    assert s.list(target="gcn") == []


def test_cubin_cache_tagged_by_toolchain(tmp_path):
    s = LocalStore(str(tmp_path))
    assert s.read_cubin(ARCH, IR_HASH, "13.3") is None
    path = s.write_cubin(ARCH, IR_HASH, "13.3", b"cubin-A")
    assert Path(path) == tmp_path / ARCH / HASH16 / f"{HASH16}.13.3.acf.cubin"
    assert s.read_cubin(ARCH, IR_HASH, "13.3") == b"cubin-A"
    # a different toolchain version must not serve the same cubin
    assert s.read_cubin(ARCH, IR_HASH, "13.4") is None
    # the cubin file is not mistaken for an artifact by list()
    assert all(not e["path"].endswith(".acf.cubin") for e in s.list())


def test_acf_is_tagged_by_toolchain_version(tmp_path):
    # All toolchain versions coexist inside the kernel's hash directory.
    s = LocalStore(str(tmp_path))
    p133 = s.write("ptx", ARCH, IR_HASH, b"acf-133", meta={}, toolchain_version="13.3")
    p134 = s.write("ptx", ARCH, IR_HASH, b"acf-134", meta={}, toolchain_version="13.4")
    assert p133 == os.path.join(str(tmp_path), ARCH, HASH16, f"{HASH16}.13.3.acf")
    assert p133 != p134
    assert s.read("ptx", ARCH, IR_HASH, toolchain_version="13.3") == b"acf-133"
    assert s.read("ptx", ARCH, IR_HASH, toolchain_version="13.4") == b"acf-134"
    # an untagged lookup must not silently serve a tagged artifact
    assert s.read("ptx", ARCH, IR_HASH) is None
    assert {e["toolchain_version"] for e in s.list()} == {"13.3", "13.4"}


def _write_flat_artifact(root, version="13.3"):
    suffix = f".{version}" if version else ""
    path = root / ARCH / f"{IR_HASH}{suffix}.acf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"flat-acf")
    Path(str(path) + ".json").write_text(json.dumps({"target": "ptx", "engine": "flat-engine"}))
    return path


@pytest.mark.parametrize("version", ["", "13.3"])
def test_read_legacy_flat_store(tmp_path, version):
    path = _write_flat_artifact(tmp_path, version)
    store = LocalStore(str(tmp_path))
    assert store.has("ptx", ARCH, IR_HASH, version)
    assert store.read("ptx", ARCH, IR_HASH, version) == b"flat-acf"
    assert store.read_meta("ptx", ARCH, IR_HASH, version)["engine"] == "flat-engine"
    entries = store.list(target="ptx", arch=ARCH)
    assert len(entries) == 1 and entries[0]["path"] == str(path)
    assert entries[0]["toolchain_version"] == (version or None)
    assert store.list(target="gcn") == []


def test_nested_artifact_wins_without_duplicating_legacy_entry(tmp_path):
    flat = _write_flat_artifact(tmp_path)
    other_version = _write_flat_artifact(tmp_path, "13.4")
    store = LocalStore(str(tmp_path))
    nested = store.write("ptx", ARCH, IR_HASH, b"nested-acf", {"engine": "nested-engine"}, "13.3")
    assert store.read("ptx", ARCH, IR_HASH, "13.3") == b"nested-acf"
    assert store.read_meta("ptx", ARCH, IR_HASH, "13.3")["engine"] == "nested-engine"
    entries = store.list()
    assert [e["path"] for e in entries] == [nested, str(other_version)]
    assert [e["meta"]["engine"] for e in entries] == ["nested-engine", "flat-engine"]
    assert flat.read_bytes() == b"flat-acf"  # Writing the new layout does not migrate or remove old files.


def test_nested_artifact_never_uses_flat_metadata_or_cubin(tmp_path):
    flat = _write_flat_artifact(tmp_path)
    Path(str(flat) + ".cubin").write_bytes(b"flat-cubin")
    store = LocalStore(str(tmp_path))
    assert store.read_cubin(ARCH, IR_HASH, "13.3") == b"flat-cubin"
    nested = store.write("ptx", ARCH, IR_HASH, b"new-acf", toolchain_version="13.3")
    Path(nested + ".json").unlink()
    assert store.read_meta("ptx", ARCH, IR_HASH, "13.3") is None
    assert store.list()[0]["meta"] == {}
    assert store.read_cubin(ARCH, IR_HASH, "13.3") is None
    store.write_cubin(ARCH, IR_HASH, "13.3", b"new-cubin")
    assert store.read_cubin(ARCH, IR_HASH, "13.3") == b"new-cubin"
    assert len(store.list()) == 1


@pytest.mark.parametrize("version", ["", "13.3"])
def test_full_hash_nested_store_remains_readable(tmp_path, version):
    suffix = f".{version}" if version else ""
    old = tmp_path / ARCH / IR_HASH / f"{IR_HASH}{suffix}.acf"
    old.parent.mkdir(parents=True)
    old.write_bytes(ACF)
    Path(str(old) + ".json").write_text(json.dumps({"target": "ptx", "ir_hash": IR_HASH, "engine": "old-engine"}))
    Path(str(old) + ".cubin").write_bytes(b"old-cubin")
    store = LocalStore(str(tmp_path))
    assert store.has("ptx", ARCH, IR_HASH, version)
    assert store.read("ptx", ARCH, IR_HASH, version) == ACF
    assert store.read_meta("ptx", ARCH, IR_HASH, version)["engine"] == "old-engine"
    assert store.list()[0]["path"] == str(old)
    assert store.list()[0]["ir_hash"] == IR_HASH
    if version:
        assert store.read_cubin(ARCH, IR_HASH, version) == b"old-cubin"

    new = store.write("ptx", ARCH, IR_HASH, b"new-acf", {"engine": "new-engine"}, version)
    assert store.read("ptx", ARCH, IR_HASH, version) == b"new-acf"
    assert store.read_meta("ptx", ARCH, IR_HASH, version)["ir_hash"] == IR_HASH
    assert [item["path"] for item in store.list()] == [new]
    Path(new + ".json").unlink()
    assert store.read_meta("ptx", ARCH, IR_HASH, version) is None
    assert store.list()[0]["meta"] == {}
    if version:
        assert store.read_cubin(ARCH, IR_HASH, version) is None
    assert old.read_bytes() == ACF


def test_full_hash_nested_precedes_flat_short_hash(tmp_path):
    flat = tmp_path / ARCH / f"{HASH16}.13.3.acf"
    flat.parent.mkdir(parents=True)
    flat.write_bytes(b"flat-short")
    Path(str(flat) + ".json").write_text(json.dumps({"target": "ptx", "ir_hash": IR_HASH}))
    store = LocalStore(str(tmp_path))
    assert store.read("ptx", ARCH, IR_HASH, "13.3") == b"flat-short"
    assert store.list()[0]["ir_hash"] == IR_HASH

    nested = tmp_path / ARCH / IR_HASH / f"{IR_HASH}.13.3.acf"
    nested.parent.mkdir()
    nested.write_bytes(b"nested-full")
    assert store.read("ptx", ARCH, IR_HASH, "13.3") == b"nested-full"
    assert store.read_meta("ptx", ARCH, IR_HASH, "13.3") is None
    assert [item["path"] for item in store.list()] == [str(nested)]


def test_sidecar_carries_full_provenance(tmp_path):
    # The search space and engine build are tuning *inputs* that the store key does NOT distinguish
    # (and the catalog defaults to "latest"), so the sidecar is the only record of which produced a
    # given ACF. Dropping these fields would make an admitted ACF unexplainable.
    s = LocalStore(str(tmp_path))
    meta = {
        "ptxas_version": "13.3",
        "engine": "compileiq",
        "engine_info": {"name": "compileiq", "version": "1.0.0.dev1", "path": "/sp/compileiq", "python": "/py"},
        "search_space": {"resolved_tag": "search-spaces-2026.05.22", "sha256": "deadbeef", "source": "cache"},
    }
    s.write("ptx", ARCH, IR_HASH, ACF, meta=meta, toolchain_version="13.3")
    got = s.read_meta("ptx", ARCH, IR_HASH, toolchain_version="13.3")
    assert got["ptxas_version"] == "13.3"
    assert got["engine_info"]["version"] == "1.0.0.dev1"
    assert got["search_space"]["resolved_tag"] == "search-spaces-2026.05.22"
    assert got["search_space"]["sha256"] == "deadbeef"


def test_default_store_root_env(monkeypatch):
    monkeypatch.setenv("COMPILE_IQ_STORE", "/tmp/ciq_store_xyz")
    assert default_store_root() == "/tmp/ciq_store_xyz"
    monkeypatch.delenv("COMPILE_IQ_STORE", raising=False)
    assert default_store_root().endswith("/.compile_iq/store")
