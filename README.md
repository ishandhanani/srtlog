# srtlog

Log analysis toolkit for SGLang distributed inference benchmarks.

## Installation

```bash
uv pip install .
```

## Usage

```bash
# Parse a run directory
srtlog parse /path/to/job_12345/

# Verbose output
srtlog parse /path/to/logs/ -v
```

Output:
```
                        4401463
╭─────────────┬───────┬───────────────────────────────╮
│ Worker Type │ Count │ Time Range                    │
├─────────────┼───────┼───────────────────────────────┤
│ Prefill     │     6 │ 16:10:37 → 17:18:27 (67m 50s) │
│ Decode      │     2 │ 16:10:46 → 17:18:35 (67m 49s) │
├─────────────┼───────┼───────────────────────────────┤
│ Total       │     8 │ 16:10:37 → 17:18:35 (67m 58s) │
╰─────────────┴───────┴───────────────────────────────╯

Cached to: /path/to/logs/cached_assets/node_metrics.parquet
```

## Python API

```python
from srtlog import NodeAnalyzer

analyzer = NodeAnalyzer()
nodes = analyzer.parse_run_logs("/path/to/job/logs")

# Filter by worker type
prefill_nodes = [n for n in nodes if n.is_prefill]
decode_nodes = [n for n in nodes if n.is_decode]
agg_nodes = [n for n in nodes if n.is_agg]

# Access batch metrics
for node in nodes:
    print(f"{node.node_name}: {len(node.batches)} batches")
    for batch in node.batches[:5]:
        print(f"  {batch.timestamp} - {batch.batch_type}")
```

## Expected Log Structure

```
run_directory/
└── logs/
    ├── node01_prefill_w0.out
    ├── node01_prefill_w1.out
    ├── node02_decode_w0.out
    ├── node02_decode_w1.out
    └── ...
```

Or for aggregated mode:
```
run_directory/
└── logs/
    ├── node01_agg_w0.out
    ├── node01_agg_w1.out
    └── ...
```

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest
```
