# Prym

A Python interface for converting results of PromQL range queries into numpy arrays or pandas DataFrames.

## Dependencies
- Python 3.6+
- requests
- numpy
- pandas

## Installation

```
pip3 install git+https://github.com/chime-experiment/prym.git
```

## Usage

```
import prym
import datetime

client = prym.Prometheus("http://prometheus-host:prometheus-query-port")

query = ('http_requests_total{job="apiserver", handler="/api/comments"}[5m]')

end = datetime.datetime.utcnow()
day = datetime.timedelta(1)
start = end - day

num_array = client.query_range(query, start, end, "5m", pandas=False)
pandas_df = client.query_range(('http_requests_total[5m]'), start, end, "5m", pandas=True)
```

```
prym.Prometheus(self, url: str)

Create a Prometheus query object.

Parameters
----------
url
    Address for the prometheus server to query.
```

```
prym.Prometheus.query_range(self, query: str, start: float or datetime, end: float or datetime, step: float, int or str, sort: func(dict)=None, pandas: bool=False)

Perform a prometheus range query. This will evaluate a PromQL query and return the results as either a numpy array, or as a pandas DataFrame.

Parameters
-----------
query
    A PromQL query string.
start, end
    The start and end of the time interval. Either as a datetime or as a floating point UNIX timestamp.
step
    The sample interval, as a prometheus duration string, or as seconds.
sort
    An optional function can be passed that generates a sort key from a metric dictionary.
pandas
    If true, return a `pd.DataFrame` with a DateTimeIndex for the times and a MultiIndex for the column labels.
```

