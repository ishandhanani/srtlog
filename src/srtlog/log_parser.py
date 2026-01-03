"""
Node analysis service for parsing .err/.out log files
"""

import logging
import os
import re

import pandas as pd

from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class NodeAnalyzer:
    """Service for analyzing node-level metrics from log files."""

    def parse_run_logs(self, run_path: str) -> list:
        """Parse all node log files in a run directory.

        Uses parquet caching to avoid re-parsing on subsequent loads.

        Args:
            run_path: Path to the run directory containing .err/.out files

        Returns:
            List of NodeMetrics objects, one per node
        """
        nodes, _ = self._parse_run_logs_impl(run_path)
        return nodes

    def parse_run_logs_with_cache_info(self, run_path: str) -> tuple[list, str | None]:
        """Parse all node log files and return cache path info.

        Args:
            run_path: Path to the run directory containing .err/.out files

        Returns:
            Tuple of (nodes, cache_path) where cache_path is the path to the parquet file
            if it was created/updated, or None if loaded from existing cache.
        """
        return self._parse_run_logs_impl(run_path)

    def _parse_run_logs_impl(self, run_path: str) -> tuple[list, str | None]:
        """Internal implementation of parse_run_logs."""
        cache_mgr = CacheManager(run_path)
        source_patterns = ["*.err", "*.out"]

        # Try to load from cache first
        if cache_mgr.is_cache_valid("node_metrics", source_patterns):
            cached_df = cache_mgr.load_from_cache("node_metrics")
            if cached_df is not None and not cached_df.empty:
                nodes = self._deserialize_node_metrics(cached_df)
                logger.info(f"Loaded {len(nodes)} nodes from cache")
                return nodes, None

        # Cache miss - parse from .err/.out files
        nodes = []
        cache_path = None

        if not os.path.exists(run_path):
            logger.error(f"Run path does not exist: {run_path}")
            return nodes, None

        total_files = 0
        parsed = 0

        for file in os.listdir(run_path):
            if (file.endswith(".err") or file.endswith(".out")) and (
                "prefill" in file or "decode" in file or "_agg_" in file
            ):
                total_files += 1
                filepath = os.path.join(run_path, file)
                node = self.parse_single_log(filepath)
                if node:
                    nodes.append(node)
                    parsed += 1

        logger.info(f"Parsed {parsed}/{total_files} worker log files from {run_path}")

        if total_files == 0:
            logger.warning(f"No worker log files found in {run_path}")

        # Save to cache
        if nodes:
            cache_df = self._serialize_node_metrics(nodes)
            cache_path = cache_mgr.save_to_cache("node_metrics", cache_df, source_patterns)

        return nodes, cache_path

    def parse_single_log(self, filepath: str):
        """Parse a single node log file."""
        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        node_info = self._extract_node_info_from_filename(filepath)
        if not node_info:
            logger.warning(f"Could not extract node info from filename: {filepath}")
            return None

        batches = []
        memory_snapshots = []
        config = {}

        try:
            with open(filepath) as f:
                for line in f:
                    # Parse prefill batch
                    batch = self._parse_prefill_batch_line(line)
                    if batch:
                        batches.append(BatchMetrics(**batch))
                        continue

                    # Parse decode batch
                    batch = self._parse_decode_batch_line(line)
                    if batch:
                        batches.append(BatchMetrics(**batch))
                        continue

                    # Parse memory
                    mem = self._parse_memory_line(line)
                    if mem:
                        memory_snapshots.append(MemoryMetrics(**mem))

                    # Extract config
                    if "--tp-size" in line:
                        tp = re.search(r"--tp-size\s+(\d+)", line)
                        dp = re.search(r"--dp-size\s+(\d+)", line)
                        ep = re.search(r"--ep-size\s+(\d+)", line)
                        if tp:
                            config["tp_size"] = int(tp.group(1))
                        if dp:
                            config["dp_size"] = int(dp.group(1))
                        if ep:
                            config["ep_size"] = int(ep.group(1))

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None

        return NodeMetrics(
            node_info=node_info,
            batches=batches,
            memory_snapshots=memory_snapshots,
            config=config,
        )

    def _serialize_node_metrics(self, nodes: list) -> pd.DataFrame:
        """Serialize NodeMetrics to DataFrame for caching."""
        rows = []

        for node in nodes:
            info = node.node_info
            cfg = node.config

            for batch in node.batches:
                rows.append({
                    "node": info.get("node", ""),
                    "worker_type": info.get("worker_type", ""),
                    "worker_id": info.get("worker_id", ""),
                    "tp_size": cfg.get("tp_size"),
                    "dp_size": cfg.get("dp_size"),
                    "ep_size": cfg.get("ep_size"),
                    "metric_type": "batch",
                    "timestamp": batch.timestamp,
                    "dp": batch.dp,
                    "tp": batch.tp,
                    "ep": batch.ep,
                    "batch_type": batch.batch_type,
                    "new_seq": batch.new_seq,
                    "new_token": batch.new_token,
                    "cached_token": batch.cached_token,
                    "token_usage": batch.token_usage,
                    "running_req": batch.running_req,
                    "queue_req": batch.queue_req,
                    "prealloc_req": batch.prealloc_req,
                    "inflight_req": batch.inflight_req,
                    "transfer_req": batch.transfer_req,
                    "preallocated_usage": batch.preallocated_usage,
                    "num_tokens": batch.num_tokens,
                    "input_throughput": batch.input_throughput,
                    "gen_throughput": batch.gen_throughput,
                })

            for mem in node.memory_snapshots:
                rows.append({
                    "node": info.get("node", ""),
                    "worker_type": info.get("worker_type", ""),
                    "worker_id": info.get("worker_id", ""),
                    "tp_size": cfg.get("tp_size"),
                    "dp_size": cfg.get("dp_size"),
                    "ep_size": cfg.get("ep_size"),
                    "metric_type": "memory",
                    "timestamp": mem.timestamp,
                    "dp": mem.dp,
                    "tp": mem.tp,
                    "ep": mem.ep,
                    "avail_mem_gb": mem.avail_mem_gb,
                    "mem_usage_gb": mem.mem_usage_gb,
                    "kv_cache_gb": mem.kv_cache_gb,
                    "kv_tokens": mem.kv_tokens,
                })

        return pd.DataFrame(rows)

    def _deserialize_node_metrics(self, df: pd.DataFrame) -> list:
        """Deserialize NodeMetrics from cached DataFrame."""
        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        nodes = []

        for (node_name, worker_type, worker_id), group in df.groupby(
            ["node", "worker_type", "worker_id"], dropna=False
        ):
            node_info = {"node": node_name, "worker_type": worker_type, "worker_id": worker_id}

            config = {}
            if not group.empty:
                row = group.iloc[0]
                if pd.notna(row.get("tp_size")):
                    config["tp_size"] = int(row["tp_size"])
                if pd.notna(row.get("dp_size")):
                    config["dp_size"] = int(row["dp_size"])
                if pd.notna(row.get("ep_size")):
                    config["ep_size"] = int(row["ep_size"])

            batch_df = group[group["metric_type"] == "batch"]
            memory_df = group[group["metric_type"] == "memory"]

            batches = []
            for row in batch_df.to_dict("records"):
                batches.append(BatchMetrics(
                    timestamp=row["timestamp"],
                    dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                    tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                    ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                    batch_type=row["batch_type"],
                    new_seq=int(row["new_seq"]) if pd.notna(row.get("new_seq")) else None,
                    new_token=int(row["new_token"]) if pd.notna(row.get("new_token")) else None,
                    cached_token=int(row["cached_token"]) if pd.notna(row.get("cached_token")) else None,
                    token_usage=row.get("token_usage") if pd.notna(row.get("token_usage")) else None,
                    running_req=int(row["running_req"]) if pd.notna(row.get("running_req")) else None,
                    queue_req=int(row["queue_req"]) if pd.notna(row.get("queue_req")) else None,
                    prealloc_req=int(row["prealloc_req"]) if pd.notna(row.get("prealloc_req")) else None,
                    inflight_req=int(row["inflight_req"]) if pd.notna(row.get("inflight_req")) else None,
                    transfer_req=int(row["transfer_req"]) if pd.notna(row.get("transfer_req")) else None,
                    preallocated_usage=row.get("preallocated_usage") if pd.notna(row.get("preallocated_usage")) else None,
                    num_tokens=int(row["num_tokens"]) if pd.notna(row.get("num_tokens")) else None,
                    input_throughput=row.get("input_throughput") if pd.notna(row.get("input_throughput")) else None,
                    gen_throughput=row.get("gen_throughput") if pd.notna(row.get("gen_throughput")) else None,
                ))

            memory_snapshots = []
            for row in memory_df.to_dict("records"):
                memory_snapshots.append(MemoryMetrics(
                    timestamp=row["timestamp"],
                    dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                    tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                    ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                    metric_type="memory",
                    avail_mem_gb=row.get("avail_mem_gb") if pd.notna(row.get("avail_mem_gb")) else None,
                    mem_usage_gb=row.get("mem_usage_gb") if pd.notna(row.get("mem_usage_gb")) else None,
                    kv_cache_gb=row.get("kv_cache_gb") if pd.notna(row.get("kv_cache_gb")) else None,
                    kv_tokens=int(row["kv_tokens"]) if pd.notna(row.get("kv_tokens")) else None,
                ))

            nodes.append(NodeMetrics(
                node_info=node_info,
                batches=batches,
                memory_snapshots=memory_snapshots,
                config=config,
            ))

        return nodes

    def _parse_dp_tp_ep_tag(self, line: str) -> tuple[int | None, int | None, int | None, str | None]:
        """Extract DP, TP, EP indices and timestamp from log line."""
        # Full format: [2025-11-04 05:31:43 DP0 TP0 EP0]
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) TP(\d+) EP(\d+)\]", line)
        if match:
            ts, dp, tp, ep = match.groups()
            return int(dp), int(tp), int(ep), ts

        # Simple: [2025-11-04 07:05:55 TP0]
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\]", line)
        if match:
            ts, tp = match.groups()
            return 0, int(tp), 0, ts

        # Pipeline: [2025-12-08 14:34:44 PP0]
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PP(\d+)\]", line)
        if match:
            ts, pp = match.groups()
            return 0, int(pp), 0, ts

        # ISO 8601 format (no DP/TP/EP): 2025-12-30T15:52:33.006497Z
        # May have ANSI codes around it like [2m2025-12-30T15:52:33.006497Z[0m
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d+Z?", line)
        if match:
            iso_ts = match.group(1)
            # Convert to standard format: 2025-12-30T15:52:33 -> 2025-12-30 15:52:33
            ts = iso_ts.replace("T", " ")
            return 0, 0, 0, ts

        return None, None, None, None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line."""
        dp, tp, ep, ts = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Prefill batch" not in line:
            return None

        metrics = {"timestamp": ts, "dp": dp, "tp": tp, "ep": ep, "batch_type": "prefill"}

        patterns = {
            "new_seq": r"#new-seq:\s*(\d+)",
            "new_token": r"#new-token:\s*(\d+)",
            "cached_token": r"#cached-token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "running_req": r"#running-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "inflight_req": r"#inflight-req:\s*(\d+)",
            "input_throughput": r"input throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                val = match.group(1)
                metrics[key] = float(val) if "." in val else int(val)

        return metrics

    def _parse_decode_batch_line(self, line: str) -> dict | None:
        """Parse decode batch log line."""
        dp, tp, ep, ts = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Decode batch" not in line:
            return None

        metrics = {"timestamp": ts, "dp": dp, "tp": tp, "ep": ep, "batch_type": "decode"}

        patterns = {
            "running_req": r"#running-req:\s*(\d+)",
            "num_tokens": r"#token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "preallocated_usage": r"pre-allocated usage:\s*([\d.]+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "transfer_req": r"#transfer-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "gen_throughput": r"gen throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                val = match.group(1)
                metrics[key] = float(val) if "." in val else int(val)

        return metrics

    def _parse_memory_line(self, line: str) -> dict | None:
        """Parse memory-related log lines."""
        dp, tp, ep, ts = self._parse_dp_tp_ep_tag(line)
        if dp is None:
            return None

        metrics = {"timestamp": ts, "dp": dp, "tp": tp, "ep": ep}

        avail = re.search(r"avail mem=([\d.]+)\s*GB", line)
        if avail:
            metrics["avail_mem_gb"] = float(avail.group(1))
            metrics["metric_type"] = "memory"

        usage = re.search(r"mem usage=([\d.]+)\s*GB", line)
        if usage:
            metrics["mem_usage_gb"] = float(usage.group(1))
            metrics["metric_type"] = "memory"

        kv = re.search(r"KV size:\s*([\d.]+)\s*GB", line)
        if kv:
            metrics["kv_cache_gb"] = float(kv.group(1))
            metrics["metric_type"] = "kv_cache"

        tokens = re.search(r"#tokens:\s*(\d+)", line)
        if tokens:
            metrics["kv_tokens"] = int(tokens.group(1))

        return metrics if "metric_type" in metrics else None

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename."""
        match = re.match(
            r"(.+)_(prefill|decode|agg|frontend)_([^.]+)\.(err|out)",
            os.path.basename(filename)
        )
        if match:
            return {
                "node": match.group(1),
                "worker_type": match.group(2),
                "worker_id": match.group(3),
            }
        return None
