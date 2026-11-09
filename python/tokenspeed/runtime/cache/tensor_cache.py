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

"""Host-backed immutable tensor results for collective-free encoders.

A model declares the cacheable boundary; the cache owns only compact tensor
copies, never request items, offsets, transport slots or model parameters.
Use it on the rank's forward thread. Models whose encoder issues collectives
must agree their miss set before skipping any collective-bearing computation.
"""

import logging
from collections.abc import Hashable

import torch
from torch import nn

from tokenspeed.runtime.cache.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class HostTensorCache(nn.Module):
    """A byte-bounded LRU of compact CPU tensor copies.

    A hit returns a private tensor on the caller's execution device. Keeping
    retained tensors on the host avoids unreserved device allocations competing
    with KV memory. This module has no parameters or persistent buffers.
    """

    def __init__(self, capacity_bytes: int):
        super().__init__()
        self.evictions = 0
        self.generation = 0
        self._cache = EmbeddingCache(capacity_bytes, on_evict=self._on_evict)
        self.hits = 0
        self.misses = 0
        self.skipped = 0

    def _on_evict(self, key, value, nbytes) -> None:
        self.evictions += 1

    def snapshot(self) -> dict[str, int]:
        """Read counters on the owner thread; clear preserves lifetime counters."""
        return {
            "capacity_bytes": self._cache.capacity_bytes,
            "current_bytes": self.current_bytes,
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "skipped": self.skipped,
            "evictions": self.evictions,
            "generation": self.generation,
        }

    @property
    def enabled(self) -> bool:
        return self._cache.capacity_bytes > 0

    @property
    def current_bytes(self) -> int:
        return self._cache.current_bytes

    def get(self, key: Hashable, *, device: torch.device) -> torch.Tensor | None:
        """Return a request-owned copy, or None when not retained."""
        if not self.enabled:
            return None
        value = self._cache.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        # copy=True also protects the retained value when the caller is on CPU.
        # A device copy belongs to the current forward stream, like encoding.
        return value.to(device=device, copy=True)

    def put(self, key: Hashable, value: torch.Tensor) -> None:
        """Retain a compact independent copy when optional storage fits."""
        # Account for bounded entry bookkeeping as well as tensor bytes.
        nbytes = value.numel() * value.element_size() + 128
        if not self.enabled or nbytes > self._cache.capacity_bytes:
            self.skipped += 1
            return
        try:
            # copy=True prevents a view retaining a much larger batch storage.
            host = value.detach().to(device="cpu", copy=True).contiguous()
            self._cache.put(key, host, nbytes)
        except (MemoryError, RuntimeError) as error:
            cpu_oom = "DefaultCPUAllocator: can't allocate memory" in str(error)
            if (
                not isinstance(error, (MemoryError, torch.OutOfMemoryError))
                and not cpu_oom
            ):
                raise
            # Only optional retention is guarded. Forward and required H2D
            # errors must propagate normally.
            self.skipped += 1
            logger.warning(
                "Skipping an encoder cache insertion: host allocation failed"
            )

    def clear(self) -> None:
        """Drop retained values; references already returned remain valid."""
        self.generation += 1
        self._cache = EmbeddingCache(
            self._cache.capacity_bytes, on_evict=self._on_evict
        )

    def _load_from_state_dict(self, *args, **kwargs):
        # State-dict reloads may replace the parent model's visual parameters.
        self.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def _apply(self, fn, recurse=True):
        # Moving/retyping the enclosing model changes its numerical contract.
        self.clear()
        return super()._apply(fn, recurse=recurse)
