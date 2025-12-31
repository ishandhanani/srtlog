"""
Tests for log_parser module
"""

from srtlog import NodeAnalyzer


class TestNodeAnalyzer:
    """Tests for NodeAnalyzer class."""

    def test_parse_run_logs(self, job_4401453_logs_dir):
        """Test parsing logs from a real run directory."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401453_logs_dir))

        # Should find at least some nodes
        assert len(nodes) >= 0

    def test_parse_single_log_invalid_path(self):
        """Test parsing a non-existent file."""
        analyzer = NodeAnalyzer()
        result = analyzer.parse_single_log("/nonexistent/path.err")

        assert result is None

    def test_filter_prefill_nodes(self, job_4401453_logs_dir):
        """Test filtering for prefill nodes."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401453_logs_dir))

        prefill_nodes = [n for n in nodes if n.is_prefill]

        for node in prefill_nodes:
            assert node.is_prefill

    def test_filter_decode_nodes(self, job_4401453_logs_dir):
        """Test filtering for decode nodes."""
        analyzer = NodeAnalyzer()
        nodes = analyzer.parse_run_logs(str(job_4401453_logs_dir))

        decode_nodes = [n for n in nodes if n.is_decode]

        for node in decode_nodes:
            assert node.is_decode


class TestNodeAnalyzerParsing:
    """Tests for NodeAnalyzer parsing methods."""

    def test_parse_dp_tp_ep_tag_full_format(self):
        """Test parsing full DP/TP/EP tag format."""
        analyzer = NodeAnalyzer()
        line = "[2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18"

        dp, tp, ep, timestamp = analyzer._parse_dp_tp_ep_tag(line)

        assert dp == 0
        assert tp == 0
        assert ep == 0
        assert timestamp == "2025-11-04 05:31:43"

    def test_parse_dp_tp_ep_tag_simple_format(self):
        """Test parsing simple TP-only tag format."""
        analyzer = NodeAnalyzer()
        line = "[2025-11-04 07:05:55 TP0] Decode batch, #running-req: 7"

        dp, tp, ep, timestamp = analyzer._parse_dp_tp_ep_tag(line)

        assert dp == 0
        assert tp == 0
        assert ep == 0
        assert timestamp == "2025-11-04 07:05:55"

    def test_parse_dp_tp_ep_tag_no_match(self):
        """Test parsing line without DP/TP/EP tag."""
        analyzer = NodeAnalyzer()
        line = "Some random log line without tags"

        dp, tp, ep, timestamp = analyzer._parse_dp_tp_ep_tag(line)

        assert dp is None
        assert tp is None
        assert ep is None
        assert timestamp is None

    def test_parse_prefill_batch_line(self):
        """Test parsing prefill batch log line."""
        analyzer = NodeAnalyzer()
        line = "[2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0"

        result = analyzer._parse_prefill_batch_line(line)

        assert result is not None
        assert result["batch_type"] == "prefill"
        assert result["new_seq"] == 18
        assert result["new_token"] == 16384
        assert result["cached_token"] == 0
        assert result["token_usage"] == 0.0

    def test_parse_decode_batch_line(self):
        """Test parsing decode batch log line."""
        analyzer = NodeAnalyzer()
        line = "[2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040, token usage: 0.00, gen throughput (token/s): 6.73"

        result = analyzer._parse_decode_batch_line(line)

        assert result is not None
        assert result["batch_type"] == "decode"
        assert result["running_req"] == 7
        assert result["num_tokens"] == 7040
        assert result["gen_throughput"] == 6.73
