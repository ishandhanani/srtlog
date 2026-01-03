"""
Tests for Engine class
"""

from srtlog import Engine, NodeAnalyzer


class TestEngine:
    """Tests for Engine class."""

    def test_engine_init_disagg(self, job_4401463_logs_dir):
        """Test Engine initialization with disaggregated logs."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        assert len(engine.prefill_nodes) > 0
        assert len(engine.decode_nodes) > 0
        assert len(engine.agg_nodes) == 0

    def test_engine_init_agg(self, job_4401453_logs_dir):
        """Test Engine initialization with aggregated logs."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401453_logs_dir))
        engine = Engine(nodes)

        assert len(engine.agg_nodes) > 0

    def test_compute_stats_prefill(self, job_4401463_logs_dir):
        """Test computing stats for prefill workers."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        summary = engine.compute_stats("input_throughput", "prefill")

        assert summary is not None
        assert summary.metric_name == "input_throughput"
        assert summary.worker_type == "prefill"
        assert summary.stats.count > 0
        assert summary.stats.mean >= 0
        assert summary.stats.median >= 0
        assert summary.stats.min <= summary.stats.max
        assert summary.stats.p25 <= summary.stats.median <= summary.stats.p75

    def test_compute_stats_decode(self, job_4401463_logs_dir):
        """Test computing stats for decode workers."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        summary = engine.compute_stats("gen_throughput", "decode")

        assert summary is not None
        assert summary.metric_name == "gen_throughput"
        assert summary.worker_type == "decode"
        assert summary.stats.count > 0

    def test_compute_all_stats(self, job_4401463_logs_dir):
        """Test computing all stats."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        all_stats = engine.compute_all_stats()

        assert "prefill" in all_stats
        assert "decode" in all_stats
        assert len(all_stats["prefill"]) > 0
        assert len(all_stats["decode"]) > 0

    def test_get_summary_table(self, job_4401463_logs_dir):
        """Test getting summary table data."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        summary = engine.get_summary_table()

        assert "prefill" in summary
        assert "decode" in summary
        # Check that the summary has metric data
        assert len(summary["prefill"]) > 0 or len(summary["decode"]) > 0

    def test_to_chart_series_node(self, job_4401463_logs_dir):
        """Test converting to chart series grouped by node."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        series = engine.to_chart_series(group_by="node")

        assert "prefill" in series
        assert "decode" in series
        assert len(series["prefill"]) > 0
        assert len(series["decode"]) > 0

    def test_to_chart_series_all(self, job_4401463_logs_dir):
        """Test converting to chart series grouped all."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401463_logs_dir))
        engine = Engine(nodes)

        series = engine.to_chart_series(group_by="all")

        assert "prefill" in series
        assert "decode" in series
        # When grouped by all, should have exactly 1 series per type
        assert len(series["prefill"]) == 1
        assert len(series["decode"]) == 1


class TestSeriesStats:
    """Tests for SeriesStats class."""

    def test_from_values(self):
        """Test creating stats from values."""
        from srtlog import SeriesStats

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = SeriesStats.from_values(values)

        assert stats is not None
        assert stats.count == 10
        assert stats.mean == 5.5
        assert stats.median == 5.5
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.p25 == 3.25
        assert stats.p75 == 7.75

    def test_from_values_with_none(self):
        """Test creating stats from values with None."""
        from srtlog import SeriesStats

        values = [1.0, None, 3.0, None, 5.0]
        stats = SeriesStats.from_values(values)

        assert stats is not None
        assert stats.count == 3
        assert stats.mean == 3.0

    def test_from_values_empty(self):
        """Test creating stats from empty values."""
        from srtlog import SeriesStats

        stats = SeriesStats.from_values([])
        assert stats is None

        stats = SeriesStats.from_values([None, None])
        assert stats is None

    def test_to_dict(self):
        """Test converting stats to dict."""
        from srtlog import SeriesStats

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = SeriesStats.from_values(values)
        d = stats.to_dict()

        assert "mean" in d
        assert "median" in d
        assert "std" in d
        assert "min" in d
        assert "max" in d
        assert "p25" in d
        assert "p75" in d
        assert "p95" in d
        assert "p99" in d
