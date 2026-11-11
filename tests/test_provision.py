# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Toolchain discovery: ptxas resolution order and the nvidia-cuda-nvcc wheel probe (no GPU).

These pin the two rules that are easy to "simplify" into a regression:
  1. an explicitly pinned ptxas is used VERBATIM, even when it is too old, and
  2. with nothing pinned, a candidate that MEETS the floor beats one that is merely found first.
"""

import ctypes
import os
from types import SimpleNamespace

import pytest

from ptx_anneal import provision


@pytest.fixture
def cuda_driver(monkeypatch):
    driver = SimpleNamespace(count=4, init_status=0, count_status=0)
    driver.cuInit = lambda flags: driver.init_status

    def device_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = driver.count
        return driver.count_status

    driver.cuDeviceGetCount = device_count
    monkeypatch.setattr(provision.ctypes, "CDLL", lambda library: driver)
    return driver


@pytest.mark.parametrize(
    ("visible", "count", "expected"),
    [
        (None, 4, ["0", "1", "2", "3"]),
        (None, 1, ["0"]),
        (None, 0, []),
        ("3, 7", 2, ["3", "7"]),
        ("GPU-abc,MIG-def", 2, ["GPU-abc", "MIG-def"]),
        ("3,-1,7", 1, ["3"]),
        ("", 0, []),
        ("-1", 0, []),
    ],
)
def test_visible_gpu_probe_respects_cuda_enumeration(cuda_driver, monkeypatch, visible, count, expected):
    cuda_driver.count = count
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert provision.visible_gpus() == expected
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible


def test_visible_gpu_probe_no_device(cuda_driver):
    cuda_driver.init_status = 100
    assert provision.visible_gpus() == []


@pytest.mark.parametrize("failure", ["init_status", "count_status"])
def test_visible_gpu_probe_reports_driver_errors(cuda_driver, failure):
    setattr(cuda_driver, failure, 999)
    with pytest.raises(RuntimeError, match="CUDA GPU discovery: .*error 999"):
        provision.visible_gpus()


def test_visible_gpu_probe_missing_driver(monkeypatch):
    def unavailable(library):
        raise OSError("missing libcuda")

    monkeypatch.setattr(provision.ctypes, "CDLL", unavailable)
    with pytest.raises(RuntimeError, match="working NVIDIA driver"):
        provision.visible_gpus()


def _fake_ptxas(tmp_path, name, version):
    """A stub that answers `--version` the way ptxas does, so ptxas_version() can parse it."""
    p = tmp_path / name
    p.write_text(f'#!/bin/bash\necho "Cuda compilation tools, release {version}, V{version}.0"\n')
    p.chmod(0o755)
    return str(p)


def test_version_parsing_and_floor():
    assert provision.min_ptxas() == provision.MIN_PTXAS
    assert provision._version_tuple("13.3") >= provision._version_tuple(provision.MIN_PTXAS)
    assert provision._version_tuple("12.9") < provision._version_tuple(provision.MIN_PTXAS)


def test_explicit_env_is_used_verbatim_even_when_too_old(tmp_path, monkeypatch):
    # The whole toolchain must agree on ONE ptxas: substituting a newer one would trade a loud
    # version error for a silent store MISS at consume time. So an explicit pin always wins.
    old = _fake_ptxas(tmp_path, "ptxas_old", "12.9")
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", old)
    monkeypatch.setattr(provision, "_wheel_ptxas", lambda: [_fake_ptxas(tmp_path, "ptxas_new", "13.3")])
    assert provision.find_ptxas() == old


def test_wheel_ptxas_preferred_over_too_old_path_ptxas(tmp_path, monkeypatch):
    # Nothing pinned: a PATH ptxas below the floor cannot apply an ACF at all, so it must not
    # shadow a new-enough wheel (this is the exact layout on a CUDA 12.x devserver).
    for var in ("TRITON_PTXAS_BLACKWELL_PATH", "TRITON_PTXAS_PATH"):
        monkeypatch.delenv(var, raising=False)
    old = _fake_ptxas(tmp_path, "ptxas_path", "12.9")
    new = _fake_ptxas(tmp_path, "ptxas_wheel", "13.3")
    monkeypatch.setattr(provision.shutil, "which", lambda _: old)
    monkeypatch.setattr(provision, "_wheel_ptxas", lambda: [new])
    assert provision.find_ptxas() == new


def test_falls_back_to_first_candidate_when_none_meet_the_floor(tmp_path, monkeypatch):
    # Nothing qualifies -> still report what we saw, so the caller can print a useful version error.
    for var in ("TRITON_PTXAS_BLACKWELL_PATH", "TRITON_PTXAS_PATH"):
        monkeypatch.delenv(var, raising=False)
    old = _fake_ptxas(tmp_path, "ptxas_path", "12.9")
    monkeypatch.setattr(provision.shutil, "which", lambda _: old)
    monkeypatch.setattr(provision, "_wheel_ptxas", list)
    assert provision.find_ptxas() == old


def test_no_ptxas_anywhere_is_none(monkeypatch):
    for var in ("TRITON_PTXAS_BLACKWELL_PATH", "TRITON_PTXAS_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(provision.shutil, "which", lambda _: None)
    monkeypatch.setattr(provision, "_wheel_ptxas", list)
    assert provision.find_ptxas() is None


def test_wheel_probe_finds_nvidia_cuda_nvcc_layout(tmp_path, monkeypatch):
    # `nvidia/cu<major>/bin/ptxas` is the nvidia-cuda-nvcc wheel layout; a pip-installed ptxas is
    # never on PATH, so without this probe such a box looks unprovisioned.
    site = tmp_path / "site-packages"
    wheel = site / "nvidia" / "cu13" / "bin"
    wheel.mkdir(parents=True)
    (wheel / "ptxas").write_text("")
    monkeypatch.setattr(provision, "_wheel_ptxas", provision._wheel_ptxas)  # ensure real impl
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"purelib": str(site), "platlib": str(site)})
    found = provision._wheel_ptxas()
    assert found == [str(wheel / "ptxas")]  # de-duplicated across purelib/platlib


def test_wheel_probe_is_empty_when_no_wheel(tmp_path, monkeypatch):
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"purelib": str(tmp_path), "platlib": str(tmp_path)})
    assert provision._wheel_ptxas() == []


def test_report_and_format_run_without_gpu_or_engine():
    # `doctor` must work on a bare install: no GPU, no engine, possibly no ptxas.
    rep = provision.report()
    assert set(rep) >= {"ptxas_path", "ptxas_version", "ptxas_min", "ptxas_ok", "engines", "gpu"}
    text = provision.format_report(rep)
    assert "ptx-anneal doctor" in text and "real tuning ready" in text
    # The engine line must not claim `tune` is broken when only the plugin registry is empty.
    if not rep["engines"]:
        assert "no plugin registered" in text and "does" in text.lower()


def test_min_ptxas_is_not_overridable_by_env(monkeypatch):
    monkeypatch.setenv("PTX_ANNEAL_MIN_PTXAS", "1.0")
    monkeypatch.setenv("MIN_PTXAS", "1.0")
    assert provision.min_ptxas() == provision.MIN_PTXAS
    assert os.environ.get("PTX_ANNEAL_MIN_PTXAS") == "1.0"  # sanity: the env really was set
