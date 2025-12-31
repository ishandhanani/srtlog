"""
Tests for models module
"""

from srtlog import BatchMetrics, MemoryMetrics, NodeMetrics


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_prefill_batch(self):
        """Test creating a prefill batch."""
        batch = BatchMetrics(
            timestamp="2025-11-04 05:31:43",
            dp=0,
            tp=0,
            ep=0,
            batch_type="prefill",
            new_token=800,
            cached_token=200,
        )

        assert batch.batch_type == "prefill"
        assert batch.new_token == 800
        assert batch.cached_token == 200

    def test_decode_batch(self):
        """Test creating a decode batch."""
        batch = BatchMetrics(
            timestamp="2025-11-04 05:31:43",
            dp=0,
            tp=0,
            ep=0,
            batch_type="decode",
            gen_throughput=123.45,
            num_tokens=1024,
        )

        assert batch.batch_type == "decode"
        assert batch.gen_throughput == 123.45
        assert batch.num_tokens == 1024


class TestMemoryMetrics:
    """Tests for MemoryMetrics dataclass."""

    def test_memory_metrics(self):
        """Test creating memory metrics."""
        mem = MemoryMetrics(
            timestamp="2025-11-04 05:31:43",
            dp=0,
            tp=0,
            ep=0,
            metric_type="memory",
            avail_mem_gb=75.11,
            mem_usage_gb=107.07,
        )

        assert mem.avail_mem_gb == 75.11
        assert mem.mem_usage_gb == 107.07


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_node_properties(self):
        """Test node property accessors."""
        node = NodeMetrics(
            node_info={
                "node": "test-node-01",
                "worker_type": "prefill",
                "worker_id": "w0",
            }
        )

        assert node.node_name == "test-node-01"
        assert node.worker_type == "prefill"
        assert node.is_prefill
        assert not node.is_decode
        assert not node.is_agg

    def test_decode_node(self):
        """Test decode node detection."""
        node = NodeMetrics(
            node_info={
                "node": "test-node-01",
                "worker_type": "decode",
                "worker_id": "w0",
            }
        )

        assert node.is_decode
        assert not node.is_prefill

    def test_agg_node(self):
        """Test aggregated node detection."""
        node = NodeMetrics(
            node_info={
                "node": "test-node-01",
                "worker_type": "agg",
                "worker_id": "w0",
            }
        )

        assert node.is_agg
        assert not node.is_prefill
        assert not node.is_decode
