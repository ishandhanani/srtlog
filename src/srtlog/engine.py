"""
Engine for computing metrics and statistics from parsed logs.
"""

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import numpy as np

from .models import NodeMetrics


@dataclass
class SeriesStats:
    """Statistics for a time series."""
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p25: float
    p75: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: list[float]) -> "SeriesStats | None":
        """Compute stats from a list of values."""
        valid = [v for v in values if v is not None]
        if not valid:
            return None
        arr = np.array(valid)
        return cls(
            count=len(valid),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "p25": round(self.p25, 2),
            "p75": round(self.p75, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class MetricSummary:
    """Summary statistics for a specific metric across workers."""
    metric_name: str
    worker_type: str
    stats: SeriesStats
    per_worker: dict[str, SeriesStats]  # worker_label -> stats


@dataclass
class ChartSeries:
    """A single series for charting."""
    label: str
    worker_type: str
    elapsed: list[float]
    values: dict[str, list]  # metric_name -> values


class Engine:
    """Engine for computing metrics and statistics."""

    METRICS = {
        "prefill": ["input_throughput", "queue_req", "inflight_req", "token_usage"],
        "decode": ["gen_throughput", "running_req", "queue_req", "transfer_req", "prealloc_req", "token_usage"],
        "agg_prefill": ["input_throughput", "queue_req", "token_usage"],
        "agg_decode": ["gen_throughput", "running_req", "queue_req", "token_usage"],
    }

    def __init__(self, nodes: list[NodeMetrics]):
        """Initialize engine with parsed nodes."""
        self.nodes = nodes
        self.prefill_nodes = [n for n in nodes if n.is_prefill]
        self.decode_nodes = [n for n in nodes if n.is_decode]
        self.agg_nodes = [n for n in nodes if n.is_agg]

    @staticmethod
    def parse_elapsed_time(timestamps: list[str]) -> list[float]:
        """Convert timestamps to elapsed seconds from first timestamp."""
        if not timestamps:
            return []
        try:
            dts = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
            start = dts[0]
            return [(dt - start).total_seconds() for dt in dts]
        except (ValueError, AttributeError):
            return list(range(len(timestamps)))

    def _detect_parallelism(self, nodes: list[NodeMetrics]) -> str:
        """Detect parallelism scheme: 'dp' or 'tp'."""
        dp_values = set()
        tp_values = set()
        for node in nodes:
            for b in node.batches:
                dp_values.add(b.dp)
                tp_values.add(b.tp)
        if len(dp_values) > 1:
            return "dp"
        if len(tp_values) > 1:
            return "tp"
        return "dp"

    def _extract_batch_values(self, batches: list, metric: str) -> list:
        """Extract metric values from batches."""
        return [getattr(b, metric, None) for b in batches]

    def compute_stats(self, metric: str, worker_type: str) -> MetricSummary | None:
        """Compute statistics for a metric across all workers of a type."""
        if worker_type == "prefill":
            nodes = self.prefill_nodes
        elif worker_type == "decode":
            nodes = self.decode_nodes
        elif worker_type == "agg":
            nodes = self.agg_nodes
        else:
            return None

        if not nodes:
            return None

        all_values = []
        per_worker = {}

        for node in nodes:
            label = f"{node.node_name}_{node.worker_type}_{node.node_info.get('worker_id', '')}"
            values = self._extract_batch_values(node.batches, metric)
            valid_values = [v for v in values if v is not None]

            if valid_values:
                all_values.extend(valid_values)
                worker_stats = SeriesStats.from_values(valid_values)
                if worker_stats:
                    per_worker[label] = worker_stats

        overall_stats = SeriesStats.from_values(all_values)
        if not overall_stats:
            return None

        return MetricSummary(
            metric_name=metric,
            worker_type=worker_type,
            stats=overall_stats,
            per_worker=per_worker,
        )

    def compute_all_stats(self) -> dict[str, list[MetricSummary]]:
        """Compute stats for all metrics by worker type."""
        results = {"prefill": [], "decode": [], "agg": []}

        for metric in self.METRICS["prefill"]:
            summary = self.compute_stats(metric, "prefill")
            if summary:
                results["prefill"].append(summary)

        for metric in self.METRICS["decode"]:
            summary = self.compute_stats(metric, "decode")
            if summary:
                results["decode"].append(summary)

        # For agg, compute both prefill and decode batch stats
        if self.agg_nodes:
            for metric in self.METRICS["agg_prefill"]:
                summary = self._compute_agg_stats(metric, "prefill")
                if summary:
                    results["agg"].append(summary)
            for metric in self.METRICS["agg_decode"]:
                summary = self._compute_agg_stats(metric, "decode")
                if summary:
                    results["agg"].append(summary)

        return results

    def _compute_agg_stats(self, metric: str, batch_type: str) -> MetricSummary | None:
        """Compute stats for agg nodes filtered by batch type."""
        if not self.agg_nodes:
            return None

        all_values = []
        per_worker = {}

        for node in self.agg_nodes:
            label = f"{node.node_name}_agg_{node.node_info.get('worker_id', '')}"
            batches = [b for b in node.batches if b.batch_type == batch_type]
            values = self._extract_batch_values(batches, metric)
            valid_values = [v for v in values if v is not None]

            if valid_values:
                all_values.extend(valid_values)
                worker_stats = SeriesStats.from_values(valid_values)
                if worker_stats:
                    per_worker[label] = worker_stats

        overall_stats = SeriesStats.from_values(all_values)
        if not overall_stats:
            return None

        return MetricSummary(
            metric_name=f"{metric} ({batch_type})",
            worker_type="agg",
            stats=overall_stats,
            per_worker=per_worker,
        )

    def to_chart_series(self, group_by: str = "node") -> dict[str, list[ChartSeries]]:
        """Convert nodes to chart series grouped as specified.

        Args:
            group_by: One of "node", "dp", or "all"

        Returns:
            Dict with keys: prefill, decode, agg_prefill, agg_decode
        """
        result = {
            "prefill": self._nodes_to_series(self.prefill_nodes, group_by),
            "decode": self._nodes_to_series(self.decode_nodes, group_by),
            "agg_prefill": self._agg_to_series(group_by, "prefill"),
            "agg_decode": self._agg_to_series(group_by, "decode"),
        }
        return result

    def _nodes_to_series(self, nodes: list[NodeMetrics], group_by: str) -> list[ChartSeries]:
        """Convert nodes to chart series."""
        if not nodes:
            return []

        if group_by == "all":
            return self._group_all(nodes)
        elif group_by == "dp":
            return self._group_by_rank(nodes)
        else:
            return self._group_by_node(nodes)

    def _group_by_node(self, nodes: list[NodeMetrics]) -> list[ChartSeries]:
        """Each node as a separate series."""
        result = []
        for node in nodes:
            if not node.batches:
                continue
            timestamps = [b.timestamp for b in node.batches]
            elapsed = self.parse_elapsed_time(timestamps)

            values = {}
            for metric in ["input_throughput", "gen_throughput", "running_req",
                          "queue_req", "inflight_req", "transfer_req",
                          "prealloc_req", "token_usage"]:
                values[metric] = [getattr(b, metric, None) for b in node.batches]

            result.append(ChartSeries(
                label=f"{node.node_name}_{node.worker_type}_{node.node_info.get('worker_id', '')}",
                worker_type=node.worker_type,
                elapsed=elapsed,
                values=values,
            ))
        return result

    def _group_by_rank(self, nodes: list[NodeMetrics]) -> list[ChartSeries]:
        """Group by DP or TP rank."""
        parallelism = self._detect_parallelism(nodes)
        groups = defaultdict(list)

        for node in nodes:
            if not node.batches:
                continue
            first_batch = node.batches[0]
            rank = first_batch.dp if parallelism == "dp" else first_batch.tp
            groups[rank].append(node)

        result = []
        for rank_idx, group_nodes in sorted(groups.items()):
            # Aggregate batches by timestamp
            batches_by_ts = defaultdict(list)
            for n in group_nodes:
                for b in n.batches:
                    batches_by_ts[b.timestamp].append(b)

            timestamps = sorted(batches_by_ts.keys())
            elapsed = self.parse_elapsed_time(timestamps)

            values = defaultdict(list)
            for ts in timestamps:
                batches = batches_by_ts[ts]
                for metric in ["input_throughput", "gen_throughput", "running_req",
                              "queue_req", "inflight_req", "transfer_req",
                              "prealloc_req", "token_usage"]:
                    vals = [getattr(b, metric, None) for b in batches]
                    valid = [v for v in vals if v is not None]
                    values[metric].append(sum(valid) / len(valid) if valid else None)

            label_prefix = "DP" if parallelism == "dp" else "TP"
            result.append(ChartSeries(
                label=f"{label_prefix}{rank_idx} ({len(group_nodes)} workers)",
                worker_type=group_nodes[0].worker_type,
                elapsed=elapsed,
                values=dict(values),
            ))
        return result

    def _group_all(self, nodes: list[NodeMetrics]) -> list[ChartSeries]:
        """Single averaged series for all nodes."""
        if not nodes:
            return []

        batches_by_ts = defaultdict(list)
        for n in nodes:
            for b in n.batches:
                batches_by_ts[b.timestamp].append(b)

        timestamps = sorted(batches_by_ts.keys())
        elapsed = self.parse_elapsed_time(timestamps)

        values = defaultdict(list)
        for ts in timestamps:
            batches = batches_by_ts[ts]
            for metric in ["input_throughput", "gen_throughput", "running_req",
                          "queue_req", "inflight_req", "transfer_req",
                          "prealloc_req", "token_usage"]:
                vals = [getattr(b, metric, None) for b in batches]
                valid = [v for v in vals if v is not None]
                values[metric].append(sum(valid) / len(valid) if valid else None)

        return [ChartSeries(
            label=f"All {nodes[0].worker_type} ({len(nodes)} workers)",
            worker_type=nodes[0].worker_type,
            elapsed=elapsed,
            values=dict(values),
        )]

    def _agg_to_series(self, group_by: str, batch_type: str) -> list[ChartSeries]:
        """Convert agg nodes to series, filtered by batch type."""
        if not self.agg_nodes:
            return []

        # Create pseudo-nodes with filtered batches
        filtered_nodes = []
        for node in self.agg_nodes:
            filtered_batches = [b for b in node.batches if b.batch_type == batch_type]
            if filtered_batches:
                # Create a lightweight wrapper
                class FilteredNode:
                    def __init__(self, orig, batches):
                        self.node_name = orig.node_name
                        self.worker_type = "agg"
                        self.node_info = orig.node_info
                        self.batches = batches
                        self.is_agg = True
                filtered_nodes.append(FilteredNode(node, filtered_batches))

        if group_by == "all":
            return self._group_all(filtered_nodes)
        elif group_by == "dp":
            return self._group_by_rank(filtered_nodes)
        else:
            return self._group_by_node(filtered_nodes)

    def get_summary_table(self) -> dict:
        """Get summary data for display."""
        stats = self.compute_all_stats()

        summary = {
            "prefill": {},
            "decode": {},
            "agg": {},
        }

        for worker_type, summaries in stats.items():
            for s in summaries:
                summary[worker_type][s.metric_name] = s.stats.to_dict()

        return summary
