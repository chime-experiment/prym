from datetime import datetime
import logging
import re
from typing import Union, Callable, Optional, Any, Tuple

import numpy as np
import pandas as pd
import requests


# Get a logger object
logger = logging.getLogger(__name__)


class Prometheus:
    """Create a Prometheus query object.

    Parameters
    ----------
    url
        Address for the prometheus server to query.
    """

    def __init__(self, url: str):
        logger.debug(f"Creating prometheus query object for {url}")
        self.url = url

    def query(self, query: str, time: Union[float, datetime]):
        """Perform a prometheus instant query."""
        raise NotImplementedError("Instant queries are not yet implemented.")

    def query_range(
        self,
        query: str,
        start: Union[float, datetime],
        end: Union[float, datetime],
        step: Union[float, int, str],
        sort: Optional[Callable[[dict], Any]] = None,
        pandas: bool = False,
    ) -> Union[Tuple[np.ndarray, list, np.ndarray], pd.DataFrame]:
        """Perform a prometheus range query.

        This will evalulate a PromQL query and return the results as either a numpy
        array, or as a pandas DataFrame.

        Parameters
        ----------
        query
            A PromQL query string.
        start, end
            The start and end of the time interval. Either as a datetime or as a
            floating point UNIX timestamp.
        step
            The sample interval, as a prometheus duration string, or as seconds.
        sort
            An optional function can be passed that generates a sort key from a
            metric dictionary.
        pandas
            If true, return a `pd.DataFrame` with a DateTimeIndex for the times and a
            MultiIndex for the column labels.

        Returns
        -------
        data
            The main dataset, either a numpy array or if `pandas` is set then a
            `pd.DataFrame`.
        metrics
            Only returned if pandas not set. A list of the metric label dictionaries.
        times
            Only returned if pandas not set. An array of the UNIX timestamps of the
            samples.
        """

        # Validate and convert the parameters into float timestamps and intervals
        st_unix = _dt_to_unix(start)
        et_unix = _dt_to_unix(end)
        step_s = _duration_to_s(step)

        logger.debug(
            f'Prometheus query="{query}" from {start} to {end} with interval {step}.'
        )

        results = self._perform_query_range(query, st_unix, et_unix, step_s)
        logger.debug(f"Received {len(results)} metrics")

        # If a sort function was passed, use it to sort the data based on the metric
        # dictionaries
        if sort:
            results = sorted(results, key=lambda r: sort(r["metric"]))

        data = self._range_query_to_numpy(results, st_unix, et_unix, step_s)

        if pandas:
            data = self._numpy_to_pandas(*data)

        return data

    def _perform_query_range(self, query, st, et, step):
        """Perform and validate a range query."""

        params = {"query": query, "start": st, "end": et, "step": step}
        r = requests.get(f"{self.url}/api/v1/query_range", params=params)

        if r.status_code != 200:
            raise RuntimeError(f"Query request failed with code {r.status_code}")

        j = r.json()

        if "status" not in j or j["status"] != "success":
            raise RuntimeError("Query was not successful.")

        if (
            "data" not in j
            or j["data"].get("resultType", None) != "matrix"
            or "result" not in j["data"]
        ):
            raise RuntimeError("Not valid prometheus formatted json.")

        return j["data"]["result"]

    @classmethod
    def _range_query_to_numpy(
        cls, results: dict, st_unix: float, et_unix: float, step_s: float
    ) -> Tuple[np.ndarray, list, np.ndarray]:
        """Take a list of results and turn it into a numpy array."""

        # Calculate the full range of timestamps we want data at. Add a small constant
        # to the end to make it inclusive if it lies exactly on an interval boundary
        times = np.arange(st_unix, et_unix + 1e-6, step_s)

        # Create the destination array and NaN fill for missing data
        data = np.zeros((len(results), len(times)), dtype=np.float64)
        data[:] = np.nan

        metrics = []

        for ii, t in enumerate(results):
            metric = t["metric"]
            metric_times, values = zip(*t["values"])

            # This identifies which slots to insert the data into. Note that it relies
            # on the fact that Prometheus produces the same grid of samples as we do in
            # here. That should be fine, and we use `np.rint` to mitigate any possible
            # rounding issues, but it's worth noting.
            inds = np.rint((np.array(metric_times) - st_unix) / step_s).astype(np.int)

            # Insert the data while converting all the string values data into floating
            # point, simply using `float` works fine as it supports all the string
            # prometheus uses for special values
            data[ii, inds] = [float(v) for v in values]

            metrics.append(metric)

        return data, metrics, times

    @classmethod
    def _numpy_to_pandas(
        cls, data: np.ndarray, metrics: list, times: np.ndarray
    ) -> pd.DataFrame:
        """Take a numpy result (with metrics and times) and convert to a DataFrame."""

        # Get the set of all the unique label names
        levels = set()
        for m in metrics:
            levels |= set(m.keys())
        levels = sorted(list(levels))
        if len(levels) == 0:
            raise RuntimeError("Queries that are constructed as pandas df need to have at least one label category in the results")

        # Get the set of label values for each metric series and turn into a multilevel
        # column index
        mt = [tuple(m.get(level, None) for level in levels) for m in metrics]
        col_index = pd.MultiIndex.from_tuples(mt, names=levels)

        return pd.DataFrame(
            data.T, columns=col_index, index=pd.to_datetime(times, unit="s")
        )


def _duration_to_s(duration: Union[float, int, str]) -> float:
    """Convert a Prometheus duration string to an interval in s.

    Parameters
    ----------
    duration
        If float or in it is assumed to be in seconds, and is passed through.
        If a str it is parsed according to prometheus rules.

    Returns
    -------
    seconds
        The number of seconds corresponding to the duration.
    """
    if not isinstance(duration, (float, int, str)):
        raise TypeError(f"Cannot convert {duration}.")

    if isinstance(duration, (float, int)):
        return float(duration)

    duration_codes = {
        "ms": 0.001,
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
        "y": 365 * 24 * 60 * 60,
    }

    # The regular expression for a single time component
    pattern = f"(\\d+)({'|'.join(duration_codes.keys())})"

    if not re.fullmatch(f"({pattern})+", duration):
        raise ValueError(f"Invalid format of duration string {duration}.")

    seconds = 0
    for match in re.finditer(pattern, duration):
        num = match.group(1)
        code = match.group(2)

        seconds += int(num) * duration_codes[code]

    return seconds


def _metric_name(mdict: dict) -> str:
    """Convert a metric-label dictionary to a string.

    Parameters
    ----------
    mdict
        Dictionary with the name and label pairs.

    Returns
    -------
    s
        The string for the metric series formatted in the standard prometheus manner.
    """

    mdict["__name__"]

    labels = [f'{key}="{value}"' for key, value in mdict.items() if key != "__name__"]

    return f"{mdict['__name__']}{{{','.join(labels)}}}"


def _dt_to_unix(dt: Union[float, datetime]) -> float:
    """Convert a datetime or float to a UNIX timestamp.

    Parameters
    ----------
    dt
        If float assume it's already a UNIX timestamp, otherwise convert a
        datetime according to the usual Python rules.

    Returns
    -------
    timestamp
        UNIX timestamp.
    """
    if not isinstance(dt, (float, datetime)):
        raise TypeError(f"dt must be a float or datetime. Got {type(dt)}")

    if isinstance(dt, float):
        return dt

    return dt.timestamp()
