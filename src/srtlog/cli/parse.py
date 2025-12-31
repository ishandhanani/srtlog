"""
srtlog parse - Parse benchmark logs to parquet format

Examples:
    srtlog parse /path/to/4401453/
    srtlog parse /path/to/logs/ -v
"""

import logging
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from ..log_parser import NodeAnalyzer

logger = logging.getLogger(__name__)
console = Console()


def add_parse_subparser(subparsers):
    """Add parse subcommand to the parser."""
    parser = subparsers.add_parser(
        "parse",
        help="Parse benchmark logs to parquet format",
        description="Parse .out/.err log files and save metrics to parquet",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to run directory containing logs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.set_defaults(func=cmd_parse)
    return parser


def get_time_range(node_list):
    """Get min/max timestamps from a list of nodes."""
    all_timestamps = []
    for n in node_list:
        for b in n.batches:
            if b.timestamp:
                all_timestamps.append(b.timestamp)
        for m in n.memory_snapshots:
            if m.timestamp:
                all_timestamps.append(m.timestamp)
    if not all_timestamps:
        return None, None
    return min(all_timestamps), max(all_timestamps)


def format_time_range(start, end):
    """Format time range as 'HH:MM:SS → HH:MM:SS (Xm Ys)'."""
    if not start or not end:
        return "-"
    # Extract just the time portion (HH:MM:SS)
    start_time = start.split(" ")[-1] if " " in start else start
    end_time = end.split(" ")[-1] if " " in end else end

    # Calculate duration
    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        duration = end_dt - start_dt
        total_secs = int(duration.total_seconds())
        mins, secs = divmod(total_secs, 60)
        duration_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    except (ValueError, TypeError):
        duration_str = ""

    return f"{start_time} → {end_time} ({duration_str})"


def cmd_parse(args):
    """Execute parse command."""
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    path = Path(args.path).resolve()
    if not path.exists():
        console.print(f"[red]Error:[/red] Path does not exist: {path}")
        return 1

    # Check if path is a run directory or contains a logs subdirectory
    logs_path = path / "logs" if (path / "logs").exists() else path

    # Parse with spinner
    analyzer = NodeAnalyzer()
    with console.status("[bold blue]Parsing log files...", spinner="dots"):
        nodes, cache_path = analyzer.parse_run_logs_with_cache_info(str(logs_path))

    if not nodes:
        console.print("[yellow]No worker nodes found in logs[/yellow]")
        return 1

    # Group by worker type
    prefill_nodes = [n for n in nodes if n.is_prefill]
    decode_nodes = [n for n in nodes if n.is_decode]
    agg_nodes = [n for n in nodes if n.is_agg]

    # Build results table
    table = Table(
        title=f"[bold]{path.name}[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Worker Type", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Time Range", justify="left")

    if prefill_nodes:
        start, end = get_time_range(prefill_nodes)
        table.add_row(
            "[blue]Prefill[/blue]",
            str(len(prefill_nodes)),
            format_time_range(start, end),
        )

    if decode_nodes:
        start, end = get_time_range(decode_nodes)
        table.add_row(
            "[green]Decode[/green]",
            str(len(decode_nodes)),
            format_time_range(start, end),
        )

    if agg_nodes:
        start, end = get_time_range(agg_nodes)
        table.add_row(
            "[magenta]Aggregated[/magenta]",
            str(len(agg_nodes)),
            format_time_range(start, end),
        )

    # Add total row
    table.add_section()
    start, end = get_time_range(nodes)
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(nodes)}[/bold]",
        f"[bold]{format_time_range(start, end)}[/bold]",
    )

    console.print()
    console.print(table)
    console.print()

    # Show cache location
    if cache_path:
        console.print(f"[dim]Cached to:[/dim] [cyan]{cache_path}[/cyan]")
    else:
        console.print("[dim]Loaded from cache[/dim]")

    return 0
