"""
Domain models for log parsing
"""

from dataclasses import dataclass, field


@dataclass
class BatchMetrics:
    """Metrics from a single batch (prefill or decode)."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    batch_type: str  # "prefill" or "decode"
    # Prefill metrics
    new_seq: int | None = None
    new_token: int | None = None
    cached_token: int | None = None
    input_throughput: float | None = None
    # Decode metrics
    gen_throughput: float | None = None
    num_tokens: int | None = None
    # Shared metrics
    token_usage: float | None = None
    running_req: int | None = None
    queue_req: int | None = None
    prealloc_req: int | None = None
    inflight_req: int | None = None
    transfer_req: int | None = None
    preallocated_usage: float | None = None


@dataclass
class MemoryMetrics:
    """Memory metrics from log lines."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    metric_type: str  # "memory" or "kv_cache"
    avail_mem_gb: float | None = None
    mem_usage_gb: float | None = None
    kv_cache_gb: float | None = None
    kv_tokens: int | None = None


@dataclass
class NodeMetrics:
    """Metrics from a single worker node."""

    node_info: dict  # node name, worker type, worker_id
    batches: list[BatchMetrics] = field(default_factory=list)
    memory_snapshots: list[MemoryMetrics] = field(default_factory=list)
    config: dict = field(default_factory=dict)  # TP/DP/EP config

    @property
    def node_name(self) -> str:
        return self.node_info.get("node", "Unknown")

    @property
    def worker_type(self) -> str:
        return self.node_info.get("worker_type", "unknown")

    @property
    def is_prefill(self) -> bool:
        return self.worker_type == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.worker_type == "decode"

    @property
    def is_agg(self) -> bool:
        return self.worker_type == "agg"
