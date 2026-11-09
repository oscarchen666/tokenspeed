# Multimodal tensor result caching

Encoder adapters may reuse the existing byte-bounded embedding LRU through
HostTensorCache. The adapter chooses cacheable computation boundaries; the
cache owns compact, immutable CPU tensor copies and returns request-owned
copies on the execution device. It holds no request, prompt offset, SHM handle,
send-ring slot, model parameter, or device-resident allocation.

TOKENSPEED_MM_ENCODER_CACHE_HOST_MB is an optional capacity in MiB per rank.
Zero disables retention. Producers can attach a SHA-256 tensor content digest
computed before device transport. The digest includes ordered tensor dtype,
shape and bytes, including all encoder layout inputs. It is separate from
multimodal pad values, routing hashes and KV prefix identities.

Call the cache on the forward thread. Adapters with collective-free replicated
encoders can decide hits locally. An encoder with TP collectives must establish
a common hit/miss plan on its actual encoder TP group before skipping a
collective-bearing operation; this cache does not perform that scheduling.

An adapter may cache per-media intermediate results while retaining the existing
request-shaped finalizer. A final-result key must include the ordered child
identities and preserve every numerical boundary. Do not change GEMM shapes,
batching, token positions, or the public wire format to manufacture cache hits.

A hit copies to the current execution device, including on CPU, so the caller
cannot mutate the retained value. Eviction only releases cache ownership.
Returned tensors remain valid. Retained values own compact storage rather than
holding slices of entire request batches. Oversize results are not retained.
Optional retention allocation failures do not invalidate an already computed
result; required computation and hit-copy failures still propagate.

Retyping or moving a model and loading its state dict clear the cache. Adapters
must also clear it whenever their streaming weight-loader modifies relevant
parameters. Caches belong to one model instance and numerical configuration.
They are not shared across workers or weight generations.

This is independent of the engine's KV prefix cache. Existing prefix-cache
flush operations do not promise an all-service multimodal flush. Owners provide
an explicit clear operation, and distributed management must acknowledge every
owner before advertising such a guarantee. Capacity covers retained payload and
entry accounting, not all active request memory or the process RSS.

## Registered ownership and runtime weight updates

An EncoderSpec can expose its result_cache. When that cache is enabled, EPD
must not look up or insert the same modality in its outer embedding cache:
otherwise it can bypass model invalidation and retain duplicate output storage.
Disabled or absent adapter caches preserve the original EPD behavior. The
capacity is immutable for a serving instance; changing it requires restart.

The runtime clears registered result caches on its forward thread immediately
before a weight-writing operation, after admission checks. A failed partial load
also leaves the cache cold. Shared image/video registrations are cleared once.
This does not make an unsuccessful model update safe to resume: existing runtime
weight-version and poisoned-state rules still govern serving.

Host snapshots expose lifetime hits, misses, skipped insertions, capacity,
retained bytes, entries, evictions and generation. Clearing increments generation
and zeros occupancy while preserving counters and already returned private copies.
The cache stays forward-thread owned; snapshots and clear are not control-thread
operations. EPD cancelled admissions still release request-owned inputs normally.
