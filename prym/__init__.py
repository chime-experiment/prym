"""Query a prometheus compatible database."""
import logging
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import urllib3

__version__ = "2023.11.0"

# Get a logger object
logger = logging.getLogger(__name__)

# Try to use orjson as a performance boost for decoding the response
try:
    import orjson

    json_decoder = orjson.loads
except ImportError:
    logger.debug(
        "Could not import orjson. Falling back to slower standard library json module.",
    )
    import json

    json_decoder = json.loads

# Try to use fastnumbers as a performance boost. This is used for converting the string
# encoded numbers to floats in the prometheus output
try:
    import fastnumbers

    float_parse = fastnumbers.try_float
except ImportError:
    logger.debug("Could not import fastnumbers. Using builtin `float` function.")
    float_parse = float


ResultsTuple = tuple[np.ndarray, list[dict[str, str]], np.ndarray]


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

    def query(
        self,
        query: str,
        time: float | datetime,
    ) -> tuple[np.ndarray, dict[str, str]]:
        """Perform a prometheus instant query."""
        raise NotImplementedError("Instant queries are not yet implemented.")

    def query_range(  # noqa: PLR0913
        self,
        query: str,
        start: float | datetime,
        end: float | datetime,
        step: float | str,
        *,
        sort: Callable[[dict], Any] | None = None,
        pandas: bool = False,
    ) -> ResultsTuple | pd.DataFrame:
        """Perform a prometheus range query.

        This will evaluate a PromQL query and return the results as either a numpy
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
            f'Prometheus query="{query}" from {start} to {end} with interval {step}.',
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

    def _perform_query_range(
        self,
        query: str,
        st: float,
        et: float,
        step: float,
    ) -> dict:
        """Perform and validate a range query."""
        params = {"query": query, "start": st, "end": et, "step": step}

        try:
            http = urllib3.PoolManager()
            r = http.request_encode_url(
                "GET",
                f"{self.url}/api/v1/query_range",
                fields=params,
            )
        except urllib3.exceptions.HTTPError as e:
            raise RuntimeError(
                "Connection failed to prometheus server {self.url}.",
            ) from e

        if r.status == 400:  # noqa: PLR2004
            raise RuntimeError(
                f"Missing or incorrect parameters ({r.status}). "
                f"Prometheus says '{r.data}'",
            )
        if r.status == 422:  # noqa: PLR2004
            raise RuntimeError(
                f"Query could not be executed ({r.status}). "
                f"Prometheus says '{r.data}'",
            )
        if r.status != 200:  # noqa: PLR2004
            raise RuntimeError(
                f"Query failed ({r.status}). Prometheus says: '{r.data}'",
            )

        j = json_decoder(r.data)

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
        cls,
        results: dict,
        st_unix: float,
        et_unix: float,
        step_s: float,
    ) -> ResultsTuple:
        """Take a list of results and turn it into a numpy array."""
        # Calculate the full range of timestamps we want data at. Add a small constant
        # to the end to make it inclusive if it lies exactly on an interval boundary
        times = np.arange(st_unix, et_unix + 1e-6 * step_s, step_s)

        # Create the destination array and NaN fill for missing data
        data = np.zeros((len(results), len(times)), dtype=np.float64)
        data[:] = np.nan

        metrics = []

        for ii, t in enumerate(results):
            metric = t["metric"]

            metric_times = [u[0] for u in t["values"]]

            # This identifies which slots to insert the data into. Note that it relies
            # on the fact that Prometheus produces the same grid of samples as we do in
            # here. That should be fine, and we use `np.rint` to mitigate any possible
            # rounding issues, but it's worth noting.
            inds = np.rint((np.array(metric_times) - st_unix) / step_s).astype(int)

            # Extract the data while converting all the string values data into floating
            # point, simply using `float` works fine as it supports all the string
            # prometheus uses for special values
            data[ii, inds] = [float_parse(u[1]) for u in t["values"]]

            metrics.append(metric)

        return data, metrics, times

    @classmethod
    def _numpy_to_pandas(
        cls,
        data: np.ndarray,
        metrics: list,
        times: np.ndarray,
    ) -> pd.DataFrame:
        """Take a numpy result (with metrics and times) and convert to a DataFrame."""
        # Get the set of all the unique label names
        levels = set()
        for m in metrics:
            levels |= set(m.keys())
        levels = sorted(levels)
        if len(levels) == 0:
            raise RuntimeError(
                "Queries that are constructed as pandas df need to have at least one "
                "label category in the results",
            )

        # Get the set of label values for each metric series and turn into a multilevel
        # column index
        mt = [tuple(m.get(level, None) for level in levels) for m in metrics]
        col_index = pd.MultiIndex.from_tuples(mt, names=levels)

        return pd.DataFrame(
            data.T,
            columns=col_index,
            index=pd.to_datetime(times, unit="s"),
        )


def _duration_to_s(duration: float | str) -> float:
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
    if not isinstance(duration, float | int | str):
        raise TypeError(f"Cannot convert {duration}.")

    if isinstance(duration, float | int):
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


def metric_name(mdict: dict[str, str]) -> str:
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
    labels = [f'{key}="{mdict[key]}"' for key in sorted(mdict) if key != "__name__"]

    return f"{mdict.get('__name__','')}{{{','.join(labels)}}}"


def _dt_to_unix(dt: float | datetime) -> float:
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
    if not isinstance(dt, float | datetime):
        raise TypeError(f"dt must be a float or datetime. Got {type(dt)}")

    if isinstance(dt, float):
        return dt

    return dt.timestamp()


def concatenate_results(results: list[ResultsTuple]) -> ResultsTuple:
    """Combine a set of separate prometheus queries.

    Parameters
    ----------
    results
        The set of results to combine. These must not have overlapping ranges of time,
        but can otherwise come in any order, with any contents.

    Returns
    -------
    combined_results
        The combined results, with sorted time and metric axes.
    """
    all_metrics = {}
    all_times = []

    # Sort the non-zero queries by the *first* timestamp of each
    results = sorted([q for q in results if len(q[2]) > 0], key=lambda x: x[2][0])

    # Extract all the unique metrics and the times for each query.
    for _, metrics, times in results:
        for m in metrics:
            mstr = metric_name(m)
            if mstr not in all_metrics:
                all_metrics[mstr] = m

        # Check they don't overlap as we don't want the complexity of merging
        if all_times and times[0] < all_times[-1][-1]:
            raise ValueError("Query time periods overlap.")
        all_times.append(times)

    # Get the final array of timestamps
    all_times = np.concatenate(all_times)

    # Get the sorted list of metrics, both as a string (for lookups) and a list of dicts
    # for the return
    mstr_list = sorted(all_metrics)
    mdict_list = [all_metrics[mstr] for mstr in mstr_list]

    # Create a nan filled array that we will fill with the actual results
    output_data = np.empty((len(mstr_list), len(all_times)), dtype=np.float64)
    output_data[:] = np.nan

    # Keep track of where in the output we current are copying to
    cur_time_ind = 0

    # Iterate over all the results
    for data, metrics, _ in results:
        num_time_in_query = data.shape[1]

        # For each metric find where it is indexed in the final result set and copy the
        # data over
        for query_metric_ind, metric in enumerate(metrics):
            mstr = metric_name(metric)

            final_metric_ind = mstr_list.index(mstr)

            output_data[
                final_metric_ind,
                cur_time_ind : (cur_time_ind + num_time_in_query),
            ] = data[query_metric_ind]

        cur_time_ind += num_time_in_query

    return output_data, mdict_list, all_times
