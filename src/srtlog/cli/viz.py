"""
srtlog viz - Generate HTML visualizations from parsed logs

Examples:
    srtlog viz /path/to/logs/ -o report.html
    srtlog viz /path/to/logs/ -o report.html --group-by dp
"""

import json
import logging
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from ..cache_manager import CacheManager
from ..engine import Engine
from ..log_parser import NodeAnalyzer
from ..models import BatchMetrics, NodeMetrics

logger = logging.getLogger(__name__)
console = Console()

CACHE_NAME = "node_metrics"
SOURCE_PATTERNS = ["logs/**/*.err", "logs/*.err", "**/*.err"]


def nodes_to_dataframe(nodes: list[NodeMetrics]) -> list[dict]:
    """Convert NodeMetrics to list of dicts for caching."""
    rows = []
    for node in nodes:
        for batch in node.batches:
            rows.append({
                "node_name": node.node_name,
                "worker_type": node.worker_type,
                "worker_id": node.node_info.get("worker_id", ""),
                "timestamp": batch.timestamp,
                "batch_type": batch.batch_type,
                "dp": batch.dp,
                "tp": batch.tp,
                "ep": batch.ep,
                "input_throughput": batch.input_throughput,
                "gen_throughput": batch.gen_throughput,
                "running_req": batch.running_req,
                "queue_req": batch.queue_req,
                "inflight_req": batch.inflight_req,
                "transfer_req": batch.transfer_req,
                "prealloc_req": batch.prealloc_req,
                "token_usage": batch.token_usage,
                "preallocated_usage": batch.preallocated_usage,
            })
    return rows


def dataframe_to_nodes(df) -> list[NodeMetrics]:
    """Reconstruct NodeMetrics from cached DataFrame."""
    nodes_dict = {}

    for _, row in df.iterrows():
        key = (row["node_name"], row["worker_type"], row.get("worker_id", ""))
        if key not in nodes_dict:
            nodes_dict[key] = {
                "node_info": {
                    "node": row["node_name"],
                    "worker_type": row["worker_type"],
                    "worker_id": row.get("worker_id", ""),
                },
                "batches": [],
            }

        batch = BatchMetrics(
            timestamp=row["timestamp"],
            batch_type=row["batch_type"],
            dp=int(row.get("dp", 0) or 0),
            tp=int(row.get("tp", 0) or 0),
            ep=int(row.get("ep", 0) or 0),
            input_throughput=row.get("input_throughput"),
            gen_throughput=row.get("gen_throughput"),
            running_req=row.get("running_req"),
            queue_req=row.get("queue_req"),
            inflight_req=row.get("inflight_req"),
            transfer_req=row.get("transfer_req"),
            prealloc_req=row.get("prealloc_req"),
            token_usage=row.get("token_usage"),
            preallocated_usage=row.get("preallocated_usage"),
        )
        nodes_dict[key]["batches"].append(batch)

    return [
        NodeMetrics(
            node_info=data["node_info"],
            batches=data["batches"],
        )
        for data in nodes_dict.values()
    ]


def add_viz_subparser(subparsers):
    """Add viz subcommand to the parser."""
    parser = subparsers.add_parser(
        "viz",
        help="Generate HTML visualizations",
        description="Generate interactive HTML report from parsed logs",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to run directory containing logs",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="report.html",
        help="Output HTML file (default: report.html)",
    )
    parser.add_argument(
        "--group-by",
        choices=["node", "dp", "all"],
        default="node",
        help="How to group workers: node (each worker), dp (by DP rank), all (single average)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=cmd_viz)
    return parser


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


def _avg(values: list) -> float | None:
    """Average non-None values."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def _extract_batch_dict(b) -> dict:
    """Extract batch metrics to dict."""
    return {
        "timestamp": b.timestamp,
        "batch_type": b.batch_type,
        "input_throughput": b.input_throughput,
        "gen_throughput": b.gen_throughput,
        "running_req": b.running_req,
        "queue_req": b.queue_req,
        "inflight_req": b.inflight_req,
        "transfer_req": b.transfer_req,
        "prealloc_req": b.prealloc_req,
        "token_usage": b.token_usage,
        "preallocated_usage": b.preallocated_usage,
    }


def _detect_parallelism(nodes: list) -> str:
    """Detect what parallelism scheme is used: 'dp' or 'tp'."""
    dp_values = set()
    tp_values = set()
    for node in nodes:
        for b in node.batches:
            dp_values.add(b.dp)
            tp_values.add(b.tp)
    # If there's more than one DP value, it's DP parallelism
    if len(dp_values) > 1:
        return "dp"
    # If only one DP value (likely 0) but multiple TP, it's TP-only
    if len(tp_values) > 1:
        return "tp"
    return "dp"  # default


def group_by_dp(nodes: list) -> list[dict]:
    """Group nodes by DP rank (or TP rank if no DP) and average their metrics."""
    parallelism = _detect_parallelism(nodes)
    groups = defaultdict(list)

    for node in nodes:
        if not node.batches:
            continue
        first_batch = node.batches[0]
        rank = first_batch.dp if parallelism == "dp" else first_batch.tp
        groups[rank].append(node)

    result = []
    for rank_idx, group_nodes in sorted(groups.items()):
        batches_by_ts = defaultdict(list)
        for n in group_nodes:
            for b in n.batches:
                batches_by_ts[b.timestamp].append(b)

        avg_batches = []
        for ts in sorted(batches_by_ts.keys()):
            batches = batches_by_ts[ts]
            avg_batches.append({
                "timestamp": ts,
                "batch_type": batches[0].batch_type,
                "input_throughput": _avg([b.input_throughput for b in batches]),
                "gen_throughput": _avg([b.gen_throughput for b in batches]),
                "running_req": _avg([b.running_req for b in batches]),
                "queue_req": _avg([b.queue_req for b in batches]),
                "inflight_req": _avg([b.inflight_req for b in batches]),
                "transfer_req": _avg([b.transfer_req for b in batches]),
                "prealloc_req": _avg([b.prealloc_req for b in batches]),
                "token_usage": _avg([b.token_usage for b in batches]),
                "preallocated_usage": _avg([b.preallocated_usage for b in batches]),
            })

        label_prefix = "DP" if parallelism == "dp" else "TP"
        result.append({
            "label": f"{label_prefix}{rank_idx} ({len(group_nodes)} workers)",
            "worker_type": group_nodes[0].worker_type,
            "batches": avg_batches,
        })

    return result


def group_all(nodes: list) -> list[dict]:
    """Aggregate all nodes into a single averaged series per worker type."""
    if not nodes:
        return []

    by_type = defaultdict(list)
    for node in nodes:
        by_type[node.worker_type].append(node)

    result = []
    for worker_type, type_nodes in by_type.items():
        batches_by_ts = defaultdict(list)
        for n in type_nodes:
            for b in n.batches:
                batches_by_ts[b.timestamp].append(b)

        avg_batches = []
        for ts in sorted(batches_by_ts.keys()):
            batches = batches_by_ts[ts]
            avg_batches.append({
                "timestamp": ts,
                "batch_type": batches[0].batch_type,
                "input_throughput": _avg([b.input_throughput for b in batches]),
                "gen_throughput": _avg([b.gen_throughput for b in batches]),
                "running_req": _avg([b.running_req for b in batches]),
                "queue_req": _avg([b.queue_req for b in batches]),
                "inflight_req": _avg([b.inflight_req for b in batches]),
                "transfer_req": _avg([b.transfer_req for b in batches]),
                "prealloc_req": _avg([b.prealloc_req for b in batches]),
                "token_usage": _avg([b.token_usage for b in batches]),
                "preallocated_usage": _avg([b.preallocated_usage for b in batches]),
            })

        result.append({
            "label": f"All {worker_type} ({len(type_nodes)} workers)",
            "worker_type": worker_type,
            "batches": avg_batches,
        })

    return result


def nodes_to_series(nodes: list, batch_type_filter: str | None = None) -> list[dict]:
    """Convert nodes to series format for plotting.

    Args:
        nodes: List of NodeMetrics
        batch_type_filter: Optional filter for batch_type ("prefill" or "decode").
                          If None, includes all batches.
    """
    result = []
    for node in nodes:
        if not node.batches:
            continue
        batches = node.batches
        if batch_type_filter:
            batches = [b for b in batches if b.batch_type == batch_type_filter]
        if not batches:
            continue
        result.append({
            "label": f"{node.node_name}_{node.worker_type}_{node.node_info.get('worker_id', '')}",
            "worker_type": node.worker_type,
            "batches": [_extract_batch_dict(b) for b in batches],
        })
    return result


def split_agg_by_batch_type(agg_nodes: list, group_by: str) -> tuple[list[dict], list[dict]]:
    """Split aggregated nodes into prefill and decode series.

    For agg mode, each worker has both prefill and decode batches.
    We separate them so we can show appropriate charts for each.

    Returns:
        Tuple of (prefill_series, decode_series)
    """
    if group_by == "dp":
        prefill_series = group_by_dp_filtered(agg_nodes, "prefill")
        decode_series = group_by_dp_filtered(agg_nodes, "decode")
    elif group_by == "all":
        prefill_series = group_all_filtered(agg_nodes, "prefill")
        decode_series = group_all_filtered(agg_nodes, "decode")
    else:  # node
        prefill_series = nodes_to_series(agg_nodes, "prefill")
        decode_series = nodes_to_series(agg_nodes, "decode")
    return prefill_series, decode_series


def group_by_dp_filtered(nodes: list, batch_type: str) -> list[dict]:
    """Group nodes by DP/TP rank, filtering to specific batch type."""
    parallelism = _detect_parallelism(nodes)
    groups = defaultdict(list)

    for node in nodes:
        filtered_batches = [b for b in node.batches if b.batch_type == batch_type]
        if not filtered_batches:
            continue
        first_batch = filtered_batches[0]
        rank = first_batch.dp if parallelism == "dp" else first_batch.tp
        groups[rank].append((node, filtered_batches))

    result = []
    for rank_idx, group_items in sorted(groups.items()):
        batches_by_ts = defaultdict(list)
        for _node, filtered_batches in group_items:
            for b in filtered_batches:
                batches_by_ts[b.timestamp].append(b)

        avg_batches = []
        for ts in sorted(batches_by_ts.keys()):
            batches = batches_by_ts[ts]
            avg_batches.append({
                "timestamp": ts,
                "batch_type": batch_type,
                "input_throughput": _avg([b.input_throughput for b in batches]),
                "gen_throughput": _avg([b.gen_throughput for b in batches]),
                "running_req": _avg([b.running_req for b in batches]),
                "queue_req": _avg([b.queue_req for b in batches]),
                "inflight_req": _avg([b.inflight_req for b in batches]),
                "transfer_req": _avg([b.transfer_req for b in batches]),
                "prealloc_req": _avg([b.prealloc_req for b in batches]),
                "token_usage": _avg([b.token_usage for b in batches]),
                "preallocated_usage": _avg([b.preallocated_usage for b in batches]),
            })

        label_prefix = "DP" if parallelism == "dp" else "TP"
        result.append({
            "label": f"{label_prefix}{rank_idx} ({len(group_items)} workers)",
            "worker_type": "agg",
            "batches": avg_batches,
        })

    return result


def group_all_filtered(nodes: list, batch_type: str) -> list[dict]:
    """Aggregate all nodes into single series, filtering to specific batch type."""
    batches_by_ts = defaultdict(list)
    node_count = 0
    for node in nodes:
        filtered_batches = [b for b in node.batches if b.batch_type == batch_type]
        if filtered_batches:
            node_count += 1
            for b in filtered_batches:
                batches_by_ts[b.timestamp].append(b)

    if not batches_by_ts:
        return []

    avg_batches = []
    for ts in sorted(batches_by_ts.keys()):
        batches = batches_by_ts[ts]
        avg_batches.append({
            "timestamp": ts,
            "batch_type": batch_type,
            "input_throughput": _avg([b.input_throughput for b in batches]),
            "gen_throughput": _avg([b.gen_throughput for b in batches]),
            "running_req": _avg([b.running_req for b in batches]),
            "queue_req": _avg([b.queue_req for b in batches]),
            "inflight_req": _avg([b.inflight_req for b in batches]),
            "transfer_req": _avg([b.transfer_req for b in batches]),
            "prealloc_req": _avg([b.prealloc_req for b in batches]),
            "token_usage": _avg([b.token_usage for b in batches]),
            "preallocated_usage": _avg([b.preallocated_usage for b in batches]),
        })

    return [{
        "label": f"All agg ({node_count} workers)",
        "worker_type": "agg",
        "batches": avg_batches,
    }]


def prepare_chart_data(series: list[dict]) -> list[dict]:
    """Prepare series data for JavaScript charts."""
    charts_data = []
    for s in series:
        timestamps = [b["timestamp"] for b in s["batches"]]
        elapsed = parse_elapsed_time(timestamps)

        charts_data.append({
            "label": s["label"],
            "worker_type": s["worker_type"],
            "elapsed": elapsed,
            "input_throughput": [b.get("input_throughput") for b in s["batches"]],
            "gen_throughput": [b.get("gen_throughput") for b in s["batches"]],
            "running_req": [b.get("running_req") for b in s["batches"]],
            "queue_req": [b.get("queue_req") for b in s["batches"]],
            "inflight_req": [b.get("inflight_req") for b in s["batches"]],
            "transfer_req": [b.get("transfer_req") for b in s["batches"]],
            "prealloc_req": [b.get("prealloc_req") for b in s["batches"]],
            "token_usage": [b.get("token_usage") for b in s["batches"]],
            "preallocated_usage": [b.get("preallocated_usage") for b in s["batches"]],
        })
    return charts_data


def generate_html(
    prefill_data: list,
    decode_data: list,
    agg_prefill_data: list,
    agg_decode_data: list,
    title: str,
    group_by: str,
    stats_summary: dict | None = None,
) -> str:
    """Generate HTML report with Plotly graphs.

    Args:
        prefill_data: Data from dedicated prefill workers (disagg mode)
        decode_data: Data from dedicated decode workers (disagg mode)
        agg_prefill_data: Prefill batches from aggregated workers
        agg_decode_data: Decode batches from aggregated workers
        title: Report title
        group_by: Grouping mode used
        stats_summary: Optional statistics summary from Engine
    """
    prefill_json = json.dumps(prefill_data)
    decode_json = json.dumps(decode_data)
    agg_prefill_json = json.dumps(agg_prefill_data)
    agg_decode_json = json.dumps(agg_decode_data)
    stats_json = json.dumps(stats_summary or {})

    # Determine what sections to show
    has_prefill = len(prefill_data) > 0
    has_decode = len(decode_data) > 0
    has_agg = len(agg_prefill_data) > 0 or len(agg_decode_data) > 0

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title} - Node Metrics</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0 0 10px 0; color: #333; }}
        h2 {{ margin: 20px 0 15px 0; color: #444; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 14px; }}
        .header-stats {{ display: flex; gap: 20px; margin-top: 15px; }}
        .header-stat {{ background: #f8f9fa; padding: 10px 15px; border-radius: 6px; }}
        .header-stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .header-stat-label {{ font-size: 12px; color: #666; }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart {{ width: 100%; height: 350px; }}
        .chart-container {{
            background: #fafafa;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }}
        .chart-stats {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            padding: 10px 0 0 0;
            border-top: 1px solid #e0e0e0;
            font-size: 11px;
            color: #555;
            flex-wrap: wrap;
        }}
        .chart-stats .stat-item {{
            display: flex;
            gap: 4px;
        }}
        .chart-stats .stat-label {{
            color: #999;
        }}
        .chart-stats .stat-value {{
            font-weight: 600;
            color: #333;
        }}
        .divider {{ border-top: 1px solid #eee; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Node Metrics Report | Grouped by: {group_by}</div>
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-value">{len(prefill_data)}</div>
                <div class="header-stat-label">Prefill Series</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value">{len(decode_data)}</div>
                <div class="header-stat-label">Decode Series</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value">{len(agg_prefill_data)}</div>
                <div class="header-stat-label">Aggregated Series</div>
            </div>
        </div>
    </div>
"""

    # Aggregated Section - shows both prefill and decode metrics for agg workers
    if has_agg:
        html += """
    <div class="section">
        <h2>Aggregated Node Metrics</h2>
        <p style="color: #666; font-size: 13px; margin-bottom: 20px;">
            Each worker handles both prefill and decode operations
        </p>

        <h3 style="color: #555; margin-bottom: 15px;">Prefill Phase</h3>
        <div class="chart-container">
            <div class="chart-title">Input Throughput</div>
            <div id="agg_prefill_throughput" class="chart"></div>
            <div id="agg_prefill_throughput_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">KV Cache Utilization (Prefill)</div>
            <div id="agg_prefill_kv" class="chart"></div>
            <div id="agg_prefill_kv_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Queued Requests (Prefill)</div>
            <div id="agg_prefill_queue" class="chart"></div>
            <div id="agg_prefill_queue_stats" class="chart-stats"></div>
        </div>

        <div class="divider"></div>

        <h3 style="color: #555; margin-bottom: 15px;">Decode Phase</h3>
        <div class="chart-container">
            <div class="chart-title">Running Requests</div>
            <div id="agg_decode_running" class="chart"></div>
            <div id="agg_decode_running_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Generation Throughput</div>
            <div id="agg_decode_throughput" class="chart"></div>
            <div id="agg_decode_throughput_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">KV Cache Utilization (Decode)</div>
            <div id="agg_decode_kv" class="chart"></div>
            <div id="agg_decode_kv_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Queued Requests (Decode)</div>
            <div id="agg_decode_queue" class="chart"></div>
            <div id="agg_decode_queue_stats" class="chart-stats"></div>
        </div>
    </div>
"""

    # Prefill Section
    if has_prefill:
        html += """
    <div class="section">
        <h2>Prefill Node Metrics</h2>
        <div class="chart-container">
            <div class="chart-title">Input Throughput</div>
            <div id="prefill_throughput" class="chart"></div>
            <div id="prefill_throughput_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Inflight Requests</div>
            <div id="prefill_inflight" class="chart"></div>
            <div id="prefill_inflight_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">KV Cache Utilization</div>
            <div id="prefill_kv" class="chart"></div>
            <div id="prefill_kv_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Queued Requests</div>
            <div id="prefill_queue" class="chart"></div>
            <div id="prefill_queue_stats" class="chart-stats"></div>
        </div>
    </div>
"""

    # Decode Section
    if has_decode:
        html += """
    <div class="section">
        <h2>Decode Node Metrics</h2>
        <div class="chart-container">
            <div class="chart-title">Running Requests</div>
            <div id="decode_running" class="chart"></div>
            <div id="decode_running_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Generation Throughput</div>
            <div id="decode_throughput" class="chart"></div>
            <div id="decode_throughput_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">KV Cache Utilization</div>
            <div id="decode_kv" class="chart"></div>
            <div id="decode_kv_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Queued Requests</div>
            <div id="decode_queue" class="chart"></div>
            <div id="decode_queue_stats" class="chart-stats"></div>
        </div>

        <div class="divider"></div>
        <h3 style="color: #555; margin-bottom: 15px;">Disaggregation Metrics</h3>
        <div class="chart-container">
            <div class="chart-title">Transfer Queue</div>
            <div id="decode_transfer" class="chart"></div>
            <div id="decode_transfer_stats" class="chart-stats"></div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Prealloc Queue</div>
            <div id="decode_prealloc" class="chart"></div>
            <div id="decode_prealloc_stats" class="chart-stats"></div>
        </div>
    </div>
"""

    # JavaScript
    html += f"""
    <script>
        const prefillData = {prefill_json};
        const decodeData = {decode_json};
        const aggPrefillData = {agg_prefill_json};
        const aggDecodeData = {agg_decode_json};
        const statsData = {stats_json};

        const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

        const defaultLayout = {{
            hovermode: 'closest',
            template: 'plotly_white',
            margin: {{ t: 20, r: 20, b: 40, l: 50 }},
            xaxis: {{ title: 'Elapsed Time (s)', showgrid: true }},
            yaxis: {{ showgrid: true }},
            legend: {{ orientation: 'h', y: -0.15 }}
        }};

        function computeStats(values) {{
            const valid = values.filter(v => v !== null && v !== undefined);
            if (valid.length === 0) return null;
            const sorted = [...valid].sort((a, b) => a - b);
            const n = sorted.length;
            const mean = valid.reduce((a, b) => a + b, 0) / n;
            const median = n % 2 === 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2 : sorted[Math.floor(n/2)];
            const min = sorted[0];
            const max = sorted[n - 1];
            const p25 = sorted[Math.floor(n * 0.25)];
            const p75 = sorted[Math.floor(n * 0.75)];
            return {{ mean, median, min, max, p25, p75, count: n }};
        }}

        function renderInlineStats(statsId, values, suffix = '') {{
            const stats = computeStats(values);
            const el = document.getElementById(statsId);
            if (!el || !stats) return;
            el.innerHTML = `
                <span class="stat-item"><span class="stat-label">Mean:</span> <span class="stat-value">${{stats.mean.toFixed(1)}}${{suffix}}</span></span>
                <span class="stat-item"><span class="stat-label">Median:</span> <span class="stat-value">${{stats.median.toFixed(1)}}${{suffix}}</span></span>
                <span class="stat-item"><span class="stat-label">Min:</span> <span class="stat-value">${{stats.min.toFixed(1)}}${{suffix}}</span></span>
                <span class="stat-item"><span class="stat-label">Max:</span> <span class="stat-value">${{stats.max.toFixed(1)}}${{suffix}}</span></span>
                <span class="stat-item"><span class="stat-label">P25:</span> <span class="stat-value">${{stats.p25.toFixed(1)}}${{suffix}}</span></span>
                <span class="stat-item"><span class="stat-label">P75:</span> <span class="stat-value">${{stats.p75.toFixed(1)}}${{suffix}}</span></span>
            `;
        }}

        function makeTraces(data, metric, yLabel) {{
            const traces = data.map((s, i) => ({{
                x: s.elapsed,
                y: s[metric],
                mode: 'lines',
                name: s.label,
                line: {{ color: colors[i % colors.length], width: 1.5 }},
                hovertemplate: '<b>' + s.label + '</b><br>Time: %{{x:.1f}}s<br>' + yLabel + ': %{{y:.2f}}<extra></extra>'
            }})).filter(t => t.y.some(v => v !== null));

            // Add horizontal average line
            if (traces.length > 0) {{
                const allVals = traces.flatMap(t => t.y).filter(v => v !== null);
                const avg = allVals.reduce((a,b) => a+b, 0) / allVals.length;
                const minX = Math.min(...traces.flatMap(t => t.x));
                const maxX = Math.max(...traces.flatMap(t => t.x));
                traces.push({{
                    x: [minX, maxX], y: [avg, avg], mode: 'lines', name: 'Avg: ' + avg.toFixed(1),
                    line: {{ color: '#000', width: 2, dash: 'dash' }},
                    hovertemplate: '<b>Average: ' + avg.toFixed(2) + '</b><extra></extra>'
                }});
            }}
            return traces;
        }}

        function makeKVTraces(data) {{
            const traces = data.map((s, i) => ({{
                x: s.elapsed,
                y: s.token_usage ? s.token_usage.map(v => v !== null ? v * 100 : null) : [],
                mode: 'lines',
                name: s.label,
                line: {{ color: colors[i % colors.length], width: 1.5 }},
                hovertemplate: '<b>' + s.label + '</b><br>Time: %{{x:.1f}}s<br>Utilization: %{{y:.1f}}%<extra></extra>'
            }})).filter(t => t.y.some(v => v !== null));

            // Add horizontal average line
            if (traces.length > 0) {{
                const allVals = traces.flatMap(t => t.y).filter(v => v !== null);
                const avg = allVals.reduce((a,b) => a+b, 0) / allVals.length;
                const minX = Math.min(...traces.flatMap(t => t.x));
                const maxX = Math.max(...traces.flatMap(t => t.x));
                traces.push({{
                    x: [minX, maxX], y: [avg, avg], mode: 'lines', name: 'Avg: ' + avg.toFixed(1) + '%',
                    line: {{ color: '#000', width: 2, dash: 'dash' }},
                    hovertemplate: '<b>Average: ' + avg.toFixed(1) + '%</b><extra></extra>'
                }});
            }}
            return traces;
        }}

        function getAllValues(data, metric) {{
            return data.flatMap(s => s[metric] || []).filter(v => v !== null);
        }}

        function getAllKVValues(data) {{
            return data.flatMap(s => (s.token_usage || []).map(v => v !== null ? v * 100 : null)).filter(v => v !== null);
        }}
"""

    # Aggregated charts - prefill and decode phases
    if has_agg:
        html += """
        // Aggregated charts - Prefill phase
        Plotly.newPlot('agg_prefill_throughput', makeTraces(aggPrefillData, 'input_throughput', 'Throughput'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Tokens/s'}});
        renderInlineStats('agg_prefill_throughput_stats', getAllValues(aggPrefillData, 'input_throughput'), ' tok/s');

        Plotly.newPlot('agg_prefill_kv', makeKVTraces(aggPrefillData),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Utilization (%)', range: [0, 100]}});
        renderInlineStats('agg_prefill_kv_stats', getAllKVValues(aggPrefillData), '%');

        Plotly.newPlot('agg_prefill_queue', makeTraces(aggPrefillData, 'queue_req', 'Queued'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('agg_prefill_queue_stats', getAllValues(aggPrefillData, 'queue_req'));

        // Aggregated charts - Decode phase
        Plotly.newPlot('agg_decode_running', makeTraces(aggDecodeData, 'running_req', 'Running'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('agg_decode_running_stats', getAllValues(aggDecodeData, 'running_req'));

        Plotly.newPlot('agg_decode_throughput', makeTraces(aggDecodeData, 'gen_throughput', 'Throughput'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Tokens/s'}});
        renderInlineStats('agg_decode_throughput_stats', getAllValues(aggDecodeData, 'gen_throughput'), ' tok/s');

        Plotly.newPlot('agg_decode_kv', makeKVTraces(aggDecodeData),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Utilization (%)', range: [0, 100]}});
        renderInlineStats('agg_decode_kv_stats', getAllKVValues(aggDecodeData), '%');

        Plotly.newPlot('agg_decode_queue', makeTraces(aggDecodeData, 'queue_req', 'Queued'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('agg_decode_queue_stats', getAllValues(aggDecodeData, 'queue_req'));
"""

    # Prefill charts
    if has_prefill:
        html += """
        // Prefill charts
        Plotly.newPlot('prefill_throughput', makeTraces(prefillData, 'input_throughput', 'Throughput'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Tokens/s'}});
        renderInlineStats('prefill_throughput_stats', getAllValues(prefillData, 'input_throughput'), ' tok/s');

        Plotly.newPlot('prefill_inflight', makeTraces(prefillData, 'inflight_req', 'Inflight'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('prefill_inflight_stats', getAllValues(prefillData, 'inflight_req'));

        Plotly.newPlot('prefill_kv', makeKVTraces(prefillData),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Utilization (%)', range: [0, 100]}});
        renderInlineStats('prefill_kv_stats', getAllKVValues(prefillData), '%');

        Plotly.newPlot('prefill_queue', makeTraces(prefillData, 'queue_req', 'Queued'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('prefill_queue_stats', getAllValues(prefillData, 'queue_req'));
"""

    # Decode charts
    if has_decode:
        html += """
        // Decode charts
        Plotly.newPlot('decode_running', makeTraces(decodeData, 'running_req', 'Running'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('decode_running_stats', getAllValues(decodeData, 'running_req'));

        Plotly.newPlot('decode_throughput', makeTraces(decodeData, 'gen_throughput', 'Throughput'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Tokens/s'}});
        renderInlineStats('decode_throughput_stats', getAllValues(decodeData, 'gen_throughput'), ' tok/s');

        Plotly.newPlot('decode_kv', makeKVTraces(decodeData),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Utilization (%)', range: [0, 100]}});
        renderInlineStats('decode_kv_stats', getAllKVValues(decodeData), '%');

        Plotly.newPlot('decode_queue', makeTraces(decodeData, 'queue_req', 'Queued'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('decode_queue_stats', getAllValues(decodeData, 'queue_req'));

        Plotly.newPlot('decode_transfer', makeTraces(decodeData, 'transfer_req', 'Transfer'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('decode_transfer_stats', getAllValues(decodeData, 'transfer_req'));

        Plotly.newPlot('decode_prealloc', makeTraces(decodeData, 'prealloc_req', 'Prealloc'),
            {...defaultLayout, yaxis: {...defaultLayout.yaxis, title: 'Requests'}});
        renderInlineStats('decode_prealloc_stats', getAllValues(decodeData, 'prealloc_req'));
"""

    html += """
    </script>
</body>
</html>"""

    return html


def cmd_viz(args):
    """Execute viz command."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    path = Path(args.path).resolve()
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        return 1

    logs_path = path / "logs" if (path / "logs").exists() else path

    # Check cache first
    cache = CacheManager(str(path))
    nodes = None

    if cache.is_cache_valid(CACHE_NAME, SOURCE_PATTERNS):
        with console.status("[bold green]Loading from cache...", spinner="dots"):
            df = cache.load_from_cache(CACHE_NAME)
            if df is not None and not df.empty:
                nodes = dataframe_to_nodes(df)
                console.print(f"[green]Loaded {len(nodes)} nodes from cache[/green]")

    # Parse if no cache or cache invalid
    if not nodes:
        analyzer = NodeAnalyzer()
        with console.status("[bold blue]Parsing log files...", spinner="dots"):
            nodes = analyzer.parse_run_logs(str(logs_path))

        if nodes:
            # Cache the results
            with console.status("[bold blue]Caching results...", spinner="dots"):
                rows = nodes_to_dataframe(nodes)
                cache.save_to_cache(CACHE_NAME, rows, SOURCE_PATTERNS)
                console.print(f"[green]Cached {len(nodes)} nodes ({len(rows)} batches)[/green]")

    if not nodes:
        console.print("[yellow]No worker nodes found in logs[/yellow]")
        return 1

    # Create engine for stats computation
    engine = Engine(nodes)

    # Split by worker type
    prefill_nodes = [n for n in nodes if n.is_prefill]
    decode_nodes = [n for n in nodes if n.is_decode]
    agg_nodes = [n for n in nodes if n.is_agg]

    # Group data based on --group-by option
    with console.status(f"[bold blue]Grouping by {args.group_by}...", spinner="dots"):
        if args.group_by == "dp":
            prefill_series = group_by_dp(prefill_nodes) if prefill_nodes else []
            decode_series = group_by_dp(decode_nodes) if decode_nodes else []
        elif args.group_by == "all":
            prefill_series = group_all(prefill_nodes) if prefill_nodes else []
            decode_series = group_all(decode_nodes) if decode_nodes else []
        else:  # node
            prefill_series = nodes_to_series(prefill_nodes)
            decode_series = nodes_to_series(decode_nodes)

        # For agg nodes, split by batch_type (prefill vs decode)
        if agg_nodes:
            agg_prefill_series, agg_decode_series = split_agg_by_batch_type(agg_nodes, args.group_by)
        else:
            agg_prefill_series, agg_decode_series = [], []

    # Prepare chart data
    prefill_data = prepare_chart_data(prefill_series)
    decode_data = prepare_chart_data(decode_series)
    agg_prefill_data = prepare_chart_data(agg_prefill_series)
    agg_decode_data = prepare_chart_data(agg_decode_series)

    if not prefill_data and not decode_data and not agg_prefill_data and not agg_decode_data:
        console.print("[yellow]No data to visualize[/yellow]")
        return 1

    # Compute statistics
    with console.status("[bold blue]Computing statistics...", spinner="dots"):
        stats_summary = engine.get_summary_table()

    # Generate HTML
    with console.status("[bold blue]Generating HTML...", spinner="dots"):
        html = generate_html(
            prefill_data, decode_data, agg_prefill_data, agg_decode_data,
            path.name, args.group_by, stats_summary
        )
        output_path = Path(args.output)
        output_path.write_text(html)

    # Summary table
    table = Table(title=f"[bold]{path.name}[/bold]", box=box.ROUNDED)
    table.add_column("Section", style="bold")
    table.add_column("Series", justify="right")
    table.add_column("Data Points", justify="right")

    if prefill_data:
        total_points = sum(len(s["elapsed"]) for s in prefill_data)
        table.add_row("[blue]Prefill[/blue]", str(len(prefill_data)), f"{total_points:,}")
    if decode_data:
        total_points = sum(len(s["elapsed"]) for s in decode_data)
        table.add_row("[green]Decode[/green]", str(len(decode_data)), f"{total_points:,}")
    if agg_prefill_data:
        total_points = sum(len(s["elapsed"]) for s in agg_prefill_data) + sum(len(s["elapsed"]) for s in agg_decode_data)
        table.add_row("[magenta]Aggregated[/magenta]", str(len(agg_prefill_data)), f"{total_points:,}")

    console.print()
    console.print(table)
    console.print()

    # Auto-open in browser
    file_url = f"file://{output_path.resolve()}"
    console.print(f"[green]Opening:[/green] [cyan]{file_url}[/cyan]")
    webbrowser.open(file_url)

    return 0
