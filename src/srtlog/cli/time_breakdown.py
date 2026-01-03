"""
srtlog time-breakdown - Visualize per-request timing breakdown from SGLang metrics

Analyzes SGLang's exported per-request metrics to show time spent in each phase:
- Queue time (waiting in scheduler)
- Prefill time (prompt processing)
- Decode time (token generation)

Usage:
    srtlog time-breakdown /path/to/metrics_dir -o breakdown.html
    srtlog time-breakdown /path/to/metrics_dir --stats-only
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from rich import box
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TimingSegment:
    """Configuration for a timing segment in the breakdown."""
    name: str
    display_name: str
    color: str
    description: str


# Define the timing segments we can extract from SGLang metrics
TIMING_SEGMENTS = [
    TimingSegment(
        name="queue_time",
        display_name="Queue Time",
        color="#FFB347",  # Orange
        description="Time waiting in scheduler queue before being scheduled",
    ),
    TimingSegment(
        name="prefill_launch_delay",
        display_name="Prefill Launch Delay",
        color="#90EE90",  # Light green
        description="Delay from forward entry to actual prefill kernel start",
    ),
    TimingSegment(
        name="prefill_time",
        display_name="Prefill Execution",
        color="#6495ED",  # Cornflower blue
        description="GPU time executing prefill (prompt processing)",
    ),
    TimingSegment(
        name="decode_time",
        display_name="Decode Time",
        color="#DDA0DD",  # Plum
        description="Time generating output tokens after first token",
    ),
    TimingSegment(
        name="other_time",
        display_name="Other/Overhead",
        color="#B8B8B8",  # Gray
        description="Remaining time (tokenization, network, etc.)",
    ),
]


def add_time_breakdown_subparser(subparsers):
    """Add time-breakdown subcommand to the parser."""
    parser = subparsers.add_parser(
        "time-breakdown",
        help="Visualize per-request timing breakdown",
        description="Analyze SGLang exported metrics to show time breakdown per request",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to metrics directory (from --export-metrics-to-file-dir)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="time_breakdown.html",
        help="Output HTML file (default: time_breakdown.html)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show statistics only, don't generate HTML",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of requests to analyze",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=cmd_time_breakdown)
    return parser


def parse_metrics_files(metrics_dir: Path, limit: Optional[int] = None) -> list[dict]:
    """Parse all SGLang metrics JSONL files in the directory."""
    requests = []

    # Find all metrics log files
    log_files = sorted(metrics_dir.glob("sglang-request-metrics-*.log"))

    if not log_files:
        # Also check for .jsonl extension
        log_files = sorted(metrics_dir.glob("*.jsonl"))

    if not log_files:
        console.print(f"[yellow]No metrics files found in {metrics_dir}[/yellow]")
        return []

    console.print(f"[blue]Found {len(log_files)} metrics files[/blue]")

    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    requests.append(data)
                    if limit and len(requests) >= limit:
                        return requests
                except json.JSONDecodeError:
                    continue

    return requests


def extract_timing_breakdown(request: dict) -> dict:
    """Extract timing breakdown from a single request's metrics."""

    # Get raw timing values (already in seconds)
    queue_time = request.get("queue_time", 0) or 0
    prefill_launch_delay = request.get("prefill_launch_delay", 0) or 0
    prefill_launch_latency = request.get("prefill_launch_latency", 0) or 0
    inference_time = request.get("inference_time", 0) or 0
    e2e_latency = request.get("e2e_latency", 0) or 0

    # Prefill time is the GPU execution time
    prefill_time = prefill_launch_latency if prefill_launch_latency > 0 else 0

    # Decode time = inference_time - (prefill_launch_delay + prefill_time)
    # This gives us the actual decode phase duration
    if inference_time > 0:
        decode_time = inference_time - prefill_launch_delay - prefill_time
        decode_time = max(0, decode_time)
    else:
        decode_time = 0

    # Calculate "other" time (overhead, tokenization, network, etc.)
    # This is e2e_latency minus the sum of: queue + inference
    # (inference already includes prefill_launch_delay + prefill + decode)
    if e2e_latency > 0 and inference_time > 0:
        other_time = e2e_latency - queue_time - inference_time
        other_time = max(0, other_time)
    elif e2e_latency > 0:
        accounted = queue_time + prefill_launch_delay + prefill_time + decode_time
        other_time = max(0, e2e_latency - accounted)
    else:
        other_time = 0

    return {
        "request_id": request.get("rid", "unknown"),
        "queue_time": queue_time,
        "prefill_launch_delay": prefill_launch_delay,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "other_time": other_time,
        "e2e_latency": e2e_latency,
        "prompt_tokens": request.get("prompt_tokens", 0) or 0,
        "completion_tokens": request.get("completion_tokens", 0) or 0,
    }


def generate_html(timing_data: list[dict], title: str) -> str:
    """Generate interactive HTML visualization."""
    import json as json_module

    # Prepare data for plotting
    request_indices = list(range(len(timing_data)))

    # Convert times to milliseconds for display
    segments_data = {}
    for segment in TIMING_SEGMENTS:
        segments_data[segment.name] = [
            d.get(segment.name, 0) * 1000 for d in timing_data
        ]

    # Calculate statistics
    stats = {}
    for segment in TIMING_SEGMENTS:
        values = segments_data[segment.name]
        if any(v > 0 for v in values):
            stats[segment.name] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }

    # Total e2e latency stats
    e2e_values = [d.get("e2e_latency", 0) * 1000 for d in timing_data]
    if any(v > 0 for v in e2e_values):
        stats["e2e_latency"] = {
            "median": float(np.median(e2e_values)),
            "mean": float(np.mean(e2e_values)),
            "p95": float(np.percentile(e2e_values, 95)),
        }

    segments_json = json_module.dumps(segments_data)
    stats_json = json_module.dumps(stats)
    segments_config = json_module.dumps([
        {"name": s.name, "display_name": s.display_name, "color": s.color, "description": s.description}
        for s in TIMING_SEGMENTS
    ])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title} - Request Time Breakdown</title>
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
        .subtitle {{ color: #666; font-size: 14px; }}
        .stats-row {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-radius: 6px;
            min-width: 120px;
        }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 11px; color: #666; margin-top: 2px; }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart {{ width: 100%; height: 500px; }}
        .descriptions {{
            margin-top: 20px;
            padding: 15px;
            background: #fafafa;
            border-radius: 6px;
        }}
        .descriptions h3 {{ margin: 0 0 10px 0; color: #555; font-size: 14px; }}
        .desc-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .desc-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        .desc-name {{ font-weight: 600; color: #333; min-width: 140px; }}
        .desc-text {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Per-Request Time Breakdown | {len(timing_data)} requests analyzed</div>
        <div class="stats-row" id="stats-row"></div>
    </div>

    <div class="section">
        <div id="chart" class="chart"></div>
        <div class="descriptions" id="descriptions"></div>
    </div>

    <script>
        const segmentsData = {segments_json};
        const stats = {stats_json};
        const segmentsConfig = {segments_config};
        const requestCount = {len(timing_data)};

        // Render stats cards
        const statsRow = document.getElementById('stats-row');
        let statsHtml = '';

        if (stats.e2e_latency) {{
            statsHtml += `
                <div class="stat-card">
                    <div class="stat-value">${{stats.e2e_latency.median.toFixed(1)}} ms</div>
                    <div class="stat-label">Median E2E Latency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${{stats.e2e_latency.p95.toFixed(1)}} ms</div>
                    <div class="stat-label">P95 E2E Latency</div>
                </div>
            `;
        }}

        for (const seg of segmentsConfig) {{
            if (stats[seg.name]) {{
                statsHtml += `
                    <div class="stat-card">
                        <div class="stat-value">${{stats[seg.name].median.toFixed(1)}} ms</div>
                        <div class="stat-label">Median ${{seg.display_name}}</div>
                    </div>
                `;
            }}
        }}

        statsHtml += `
            <div class="stat-card">
                <div class="stat-value">${{requestCount}}</div>
                <div class="stat-label">Total Requests</div>
            </div>
        `;
        statsRow.innerHTML = statsHtml;

        // Create stacked bar chart
        const traces = [];
        const requestIndices = Array.from({{length: requestCount}}, (_, i) => i);

        for (const seg of segmentsConfig) {{
            const values = segmentsData[seg.name] || [];
            if (values.some(v => v > 0)) {{
                traces.push({{
                    x: requestIndices,
                    y: values,
                    name: seg.display_name,
                    type: 'bar',
                    marker: {{ color: seg.color }},
                    hovertemplate: '<b>Request %{{x}}</b><br>' + seg.display_name + ': %{{y:.2f}} ms<extra></extra>'
                }});
            }}
        }}

        const layout = {{
            barmode: 'stack',
            title: {{
                text: 'Time Breakdown per Request',
                font: {{ size: 16 }}
            }},
            xaxis: {{
                title: 'Request Index',
                showgrid: true
            }},
            yaxis: {{
                title: 'Time (milliseconds)',
                showgrid: true
            }},
            hovermode: 'closest',
            legend: {{
                orientation: 'h',
                y: -0.15
            }},
            margin: {{ t: 50, b: 80 }}
        }};

        Plotly.newPlot('chart', traces, layout, {{responsive: true}});

        // Render descriptions
        let descHtml = '<h3>Segment Descriptions</h3>';
        for (const seg of segmentsConfig) {{
            if (segmentsData[seg.name] && segmentsData[seg.name].some(v => v > 0)) {{
                descHtml += `
                    <div class="desc-item">
                        <div class="desc-color" style="background: ${{seg.color}}"></div>
                        <span class="desc-name">${{seg.display_name}}</span>
                        <span class="desc-text">${{seg.description}}</span>
                    </div>
                `;
            }}
        }}
        document.getElementById('descriptions').innerHTML = descHtml;
    </script>
</body>
</html>
"""
    return html


def show_statistics(timing_data: list[dict]):
    """Display statistics table."""
    if not timing_data:
        console.print("[yellow]No data to analyze[/yellow]")
        return

    table = Table(title="Timing Statistics (milliseconds)", box=box.ROUNDED)
    table.add_column("Segment", style="bold")
    table.add_column("Median", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Max", justify="right")

    for segment in TIMING_SEGMENTS:
        values = [d.get(segment.name, 0) * 1000 for d in timing_data]
        if any(v > 0 for v in values):
            table.add_row(
                segment.display_name,
                f"{np.median(values):.1f}",
                f"{np.mean(values):.1f}",
                f"{np.percentile(values, 95):.1f}",
                f"{np.max(values):.1f}",
            )

    # Add e2e latency row
    e2e_values = [d.get("e2e_latency", 0) * 1000 for d in timing_data]
    if any(v > 0 for v in e2e_values):
        table.add_row(
            "[bold]E2E Latency[/bold]",
            f"[bold]{np.median(e2e_values):.1f}[/bold]",
            f"[bold]{np.mean(e2e_values):.1f}[/bold]",
            f"[bold]{np.percentile(e2e_values, 95):.1f}[/bold]",
            f"[bold]{np.max(e2e_values):.1f}[/bold]",
        )

    console.print()
    console.print(table)
    console.print()

    # Token statistics
    prompt_tokens = [d.get("prompt_tokens", 0) for d in timing_data]
    completion_tokens = [d.get("completion_tokens", 0) for d in timing_data]

    if any(t > 0 for t in prompt_tokens) or any(t > 0 for t in completion_tokens):
        token_table = Table(title="Token Statistics", box=box.ROUNDED)
        token_table.add_column("Metric", style="bold")
        token_table.add_column("Median", justify="right")
        token_table.add_column("Mean", justify="right")
        token_table.add_column("Max", justify="right")

        if any(t > 0 for t in prompt_tokens):
            token_table.add_row(
                "Prompt Tokens",
                f"{np.median(prompt_tokens):.0f}",
                f"{np.mean(prompt_tokens):.0f}",
                f"{np.max(prompt_tokens):.0f}",
            )
        if any(t > 0 for t in completion_tokens):
            token_table.add_row(
                "Completion Tokens",
                f"{np.median(completion_tokens):.0f}",
                f"{np.mean(completion_tokens):.0f}",
                f"{np.max(completion_tokens):.0f}",
            )

        console.print(token_table)
        console.print()


def cmd_time_breakdown(args):
    """Execute time-breakdown command."""
    import webbrowser

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    metrics_path = Path(args.path).resolve()
    if not metrics_path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {metrics_path}")
        return 1

    # Parse metrics files
    with console.status("[bold blue]Parsing metrics files...", spinner="dots"):
        requests = parse_metrics_files(metrics_path, limit=args.limit)

    if not requests:
        console.print("[yellow]No request metrics found[/yellow]")
        return 1

    console.print(f"[green]Parsed {len(requests)} requests[/green]")

    # Extract timing breakdown for each request
    with console.status("[bold blue]Extracting timing breakdown...", spinner="dots"):
        timing_data = [extract_timing_breakdown(req) for req in requests]

    # Show statistics
    show_statistics(timing_data)

    # Generate HTML unless stats-only
    if not args.stats_only:
        with console.status("[bold blue]Generating HTML...", spinner="dots"):
            html = generate_html(timing_data, metrics_path.name)
            output_path = Path(args.output)
            output_path.write_text(html)

        file_url = f"file://{output_path.resolve()}"
        console.print(f"[green]Saved:[/green] {output_path}")
        console.print(f"[green]Opening:[/green] [cyan]{file_url}[/cyan]")
        webbrowser.open(file_url)

    return 0
