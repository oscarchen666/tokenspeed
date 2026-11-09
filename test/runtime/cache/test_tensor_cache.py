# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Behavior of immutable, bounded encoder tensor retention."""

import pytest
import torch

from tokenspeed.runtime.cache.tensor_cache import HostTensorCache
from tokenspeed.runtime.multimodal.hash import tensor_content_digest


def test_results_are_private_compact_copies():
    cache = HostTensorCache(1024)
    batch = torch.arange(1000)
    original = batch[5:8]
    cache.put(b"a", original)
    original.fill_(0)
    hit = cache.get(b"a", device=torch.device("cpu"))
    assert torch.equal(hit, torch.tensor([5, 6, 7]))
    hit.fill_(8)
    assert torch.equal(
        cache.get(b"a", device=torch.device("cpu")), torch.tensor([5, 6, 7])
    )
    assert cache.current_bytes == 3 * 8 + 128


def test_lru_budget_and_oversize_admission():
    cache = HostTensorCache(288)
    for key in (b"a", b"b"):
        cache.put(key, torch.ones(2, dtype=torch.int64))
    assert cache.get(b"a", device=torch.device("cpu")) is not None
    cache.put(b"c", torch.zeros(2, dtype=torch.int64))
    assert cache.get(b"b", device=torch.device("cpu")) is None
    cache.put(b"oversize", torch.zeros(1000))
    assert cache.get(b"a", device=torch.device("cpu")) is not None
    assert cache.current_bytes <= 288


def test_disabled_cache_has_no_retained_result():
    cache = HostTensorCache(0)
    cache.put(b"a", torch.ones(1))
    assert cache.get(b"a", device=torch.device("cpu")) is None
    assert cache.current_bytes == 0


def test_clear_and_dtype_change_invalidate_without_invalidating_readers():
    cache = HostTensorCache(1024)
    cache.put("a", torch.ones(3))
    reader = cache.get("a", device=torch.device("cpu"))
    cache.clear()
    assert cache.get("a", device=torch.device("cpu")) is None
    assert torch.equal(reader, torch.ones(3))
    cache.put("a", torch.ones(3))
    cache.to(dtype=torch.float64)
    assert cache.get("a", device=torch.device("cpu")) is None
    assert not cache.state_dict()


def test_negative_capacity_is_refused():
    with pytest.raises(ValueError, match="capacity_bytes"):
        HostTensorCache(-1)


def test_digest_includes_shape_dtype_grid_and_contents():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    grid = torch.tensor([[1, 3, 4]], dtype=torch.int64)
    digest = tensor_content_digest(tensor, grid)
    assert digest == tensor_content_digest(tensor.clone(), grid.clone())
    assert digest != tensor_content_digest(tensor.reshape(4, 3), grid)
    assert digest != tensor_content_digest(tensor.to(torch.int32), grid)
    assert digest != tensor_content_digest(tensor, grid + 1)
    assert digest != tensor_content_digest(tensor + 1, grid)


def test_digest_handles_noncontiguous_and_empty_tensors():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert tensor_content_digest(tensor) == tensor_content_digest(tensor.contiguous())
    assert len(tensor_content_digest(torch.empty(0, 3))) == 32
    assert len(tensor_content_digest(torch.tensor(1.0))) == 32


def test_state_dict_reload_drops_retained_values():
    cache = HostTensorCache(1024)
    cache.put("a", torch.ones(3))
    cache.load_state_dict({})
    assert cache.get("a", device=torch.device("cpu")) is None


def test_content_digest_has_a_stable_producer_wire_format():
    feature = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    grid = torch.tensor([[1, 1, 2]], dtype=torch.int64)
    assert tensor_content_digest(feature, grid).hex() == (
        "a76c1d26602ba3c3c84e9cda86047a65b09b2a25d5f0106aaaf4b4561edadfb6"
    )


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("injected host allocation failure"),
        RuntimeError("DefaultCPUAllocator: can't allocate memory: injected failure"),
        torch.OutOfMemoryError("injected optional allocation failure"),
    ],
)
def test_optional_allocation_failure_preserves_the_current_result(monkeypatch, error):
    cache = HostTensorCache(1024)
    result = torch.tensor([1, 2, 3])

    def fail_copy(*args, **kwargs):
        raise error

    # No real exhaustion is induced in a shared test environment.
    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    cache.put("a", result)
    assert torch.equal(result, torch.tensor([1, 2, 3]))
    assert cache.current_bytes == 0
    assert cache.skipped == 1


def test_nonallocation_copy_errors_are_not_suppressed(monkeypatch):
    cache = HostTensorCache(1024)

    def fail_copy(*args, **kwargs):
        raise RuntimeError("invalid tensor operation")

    monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    with pytest.raises(RuntimeError, match="invalid tensor operation"):
        cache.put("a", torch.ones(3))


def test_lifetime_counters_and_generation_survive_clear():
    cache = HostTensorCache(144)
    cache.put("a", torch.ones(2, dtype=torch.int64))
    cache.put("b", torch.ones(2, dtype=torch.int64))
    assert cache.get("a", device=torch.device("cpu")) is None
    assert cache.get("b", device=torch.device("cpu")) is not None
    before = cache.snapshot()
    assert before["evictions"] == 1 and before["entries"] == 1
    cache.clear()
    after = cache.snapshot()
    assert after["entries"] == after["current_bytes"] == 0
    assert after["generation"] == before["generation"] + 1
    assert after["hits"] == before["hits"] == 1
    assert after["misses"] == before["misses"] == 1
