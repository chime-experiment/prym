import numpy as np
import prym
import pytest

# Saved values from a query against an actual prometheus instance
# The query parameters...
query = (
    'delta(kotekan_rfiframedrop_dropped_frame_total{freq_id=~"300|100"}[2m]) / '
    'delta(kotekan_rfiframedrop_dropped_frame_total{freq_id=~"300|100"}[2m])'
)
start = 1700590237.0
end = start + 1200
step = 120

# The actual returned json result used for mocking the response
result = b'{"status":"success","data":{"resultType":"matrix","result":[{"metric":{"freq_id":"100","instance":"csCg9","job":"kotekan","stage_name":"/rfi_stage2/rfi1"},"values":[[1700590237,"1"],[1700590357,"1"],[1700590477,"1"],[1700590597,"1"],[1700590717,"1"],[1700590837,"1"],[1700590957,"1"],[1700591077,"1"],[1700591197,"1"],[1700591317,"1"],[1700591437,"1"]]},{"metric":{"freq_id":"300","instance":"cnCg7","job":"kotekan","stage_name":"/rfi_stage2/rfi2"},"values":[[1700590237,"NaN"],[1700590357,"NaN"],[1700590477,"NaN"],[1700590597,"1"],[1700590717,"1"],[1700590837,"1"],[1700590957,"1"],[1700591077,"NaN"],[1700591197,"NaN"],[1700591317,"NaN"],[1700591437,"NaN"]]}]}}'

# And the parsed return values...
test_values = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
    ],
    dtype=np.float64,
)

test_metrics = [
    {
        "freq_id": "100",
        "instance": "csCg9",
        "job": "kotekan",
        "stage_name": "/rfi_stage2/rfi1",
    },
    {
        "freq_id": "300",
        "instance": "cnCg7",
        "job": "kotekan",
        "stage_name": "/rfi_stage2/rfi2",
    },
]


@pytest.fixture
def full_query(httpserver):
    httpserver.expect_request(
        "/api/v1/query_range",
        query_string={
            "start": str(start),
            "end": str(end),
            "step": str(step),
            "query": query,
        },
    ).respond_with_data(result)

    return httpserver.url_for("/")


def test_query(full_query):
    """Test a query returning actual results."""
    p = prym.Prometheus(full_query)

    data, metrics, times = p.query_range(query, start=start, end=end, step="2m")

    assert len(metrics) == 2
    assert len(times) == 11

    assert data.shape == (2, 11)

    assert np.array_equal(data, test_values, equal_nan=True)
    print(metrics)

    assert metrics[0] == test_metrics[0]
    assert metrics[1] == test_metrics[1]


@pytest.fixture
def empty_query(httpserver):
    result = b'{"status":"success","data":{"resultType":"matrix","result":[]}}'

    httpserver.expect_request("/api/v1/query_range").respond_with_data(result)

    return httpserver.url_for("/")


def test_empty_query(empty_query):
    """Test the returned output for a query with no matching metrics."""
    p = prym.Prometheus(empty_query)

    data, metrics, times = p.query_range(query, start=start, end=end, step="2m")

    assert len(metrics) == 0
    assert len(times) == 11

    assert data.shape == (0, 11)
