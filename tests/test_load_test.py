"""Le test de charge lit les bons percentiles : la P95 voit la queue lente que la médiane cache."""
from load_test import summarize


def test_p95_sees_the_slow_tail():
    latencies = [10.0] * 90 + [500.0] * 10  # 10 % de requêtes lentes
    result = summarize(latencies, errors=0, seconds=10)
    assert result["p50"] == 10.0          # la médiane ne voit rien
    assert result["p95"] == 500.0         # la P95 voit la queue
    assert result["rps"] == 10.0
