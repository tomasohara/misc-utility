#! /usr/bin/env python
#
# Compute cross correlation of two sequences.
#
# TODO:
# - Show example.
# - Putting timestamp interpretation in separate module.
# - Rename 'timestamp' to 'datetime_obj' or 'timestamp_secs' depending on usage.
#

"""Computes cross correlation"""

import datetime
import sys

# Note: python 3.6+ format strings are used
assert((sys.version_info.major >= 3) and (sys.version_info.minor >= 6))

import numpy as np
import scipy.signal as ss

## TODO: import re
from main import Main
import debug
import system
## TODO: replace tpo_common with system and debug
import tpo_common as tpo

COL1 = "col1"
COL2 = "col2"
## TODO: FILENAME = "filename"
TIMESTAMP = "timestamp"
COMPUTE_STATS = "stats"
DELIM = system.getenv_text("DELIM", ",")
NORMALIZE = "normalize"
UNNORMALIZED_CCF = system.getenv_bool("UNNORMALIZED_CCF", False)
NORMALIZE_CCF = not UNNORMALIZED_CCF
# Note: Seuqence 'extension' is for endpoints and 'expansion' for gaps.
EXTEND_SEQUENCES = system.getenv_bool("EXTEND_SEQUENCES", False)
RELATIVIZE_TIMESTAMPS = system.getenv_bool("RELATIVIZE_TIMESTAMPS", not EXTEND_SEQUENCES)
ONE_DAY = (60 * 60 * 24)        # seconds in a day (86,400)

#-------------------------------------------------------------------------------

def safe_float(text, default_value=0.0):
    """Convert TEXT to floating point value using DEFAULT if invalid"""
    value = default_value
    try:
        value = float(text)
    except ValueError:
        debug.trace_fmt(7, "Bad floating point value: {t}", t=text)
    return value


EPOCH_TIMESTAMP_OBJ = datetime.datetime(1970, 1, 1)

def to_timestamp_obj(text):
    """Convert TEXT into timestamp object (i.e., datetime.datetime)"""
    # Note: datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
    # EX: to_timestamp_obj("2/23/2019") => datetime.datetime(2019, 2, 23, 0, 0)
    # EX: to_timestamp_obj("2019-2-23") => datetime.datetime(2019, 2, 23, 0, 0)
    # TODO: track down clients who are passing in text with newlines
    timestamp_obj = EPOCH_TIMESTAMP_OBJ
    resolved = False
    text_proper = text.strip()
    for date_format in ["%Y-%m-%d", "%m/%d/%Y"]:
        try:
            timestamp_obj = datetime.datetime.strptime(text_proper, date_format)
            resolved = True
            break
        except (TypeError, ValueError):
            pass
    if not resolved:
        debug.trace_fmt(2, "Unable to convert timestamp text '{t}'", t=text_proper)
    debug.trace_fmtd(7, "to_timestamp_obj({t}) => {ts}", t=text, ts=timestamp_obj)
    return timestamp_obj


def timestamp_obj_to_secs(timestamp_obj):
    """Returns seconds for TIMESTAMP_OBJ object with respect to system epoch(1/1/1970)"""
    # EX: timestamp_obj_to_secs(datetime.datetime(1970,1,2)) => 86400
    secs = -1
    try:
        secs = int((timestamp_obj - EPOCH_TIMESTAMP_OBJ).total_seconds())    
    except (ValueError, ValueError):
        debug.trace_fmt(3, "Unable to convert timestamp object {ts}", ts=timestamp_obj)
    debug.trace_fmtd(7, "timestamp_obj_to_secs({ts}) => {s}", ts=timestamp_obj, s=secs)
    return secs


def timestamp_text_to_secs(text):
    """Convert timestamp TEXT into integral time based on seconds since start of 1970"""
    # EX: timestamp_text_to_secs("1/11/1970") => 864000
    secs = timestamp_obj_to_secs(to_timestamp_obj(text))
    debug.trace_fmtd(7, "timestamp_text_to_secs({t}) => {s}", t=text, s=secs)
    return secs


def extract_timestamped_values(vector, numeric=False, timestamp_delim=None, max_timestamp_secs=None):
    """Remove timestamps from time-ordered vector and return as separate vector, along with values, converted to floating point if NUMERIC.
    Note: ignores values after MAX_TIMESTAMP_SECS and uses TIMESTAMP_DELIM for older format encoding with values suffixed by timestamp"""
    # EX: extract_timestamped_values([("1/1/2001", "5"), ("1/2/2001", "10.5"), ("1/3/2001", "15")]) => [978307200, 978393600, 978480000], [5, 10.5, 15]
    # EX: extract_timestamped_values(["5@1/1/2001", "10.5@1/2/2001", "15@1/3/2001"]) => [978307200, 978393600, 978480000], [5, 10.5, 15]
    debug.trace_fmtd(5, "extract_timestamped_values({v}, [num={n}, del={tsd}, maxts={mts})",
                     v=vector, n=numeric, tsd=timestamp_delim, mts=max_timestamp_secs)
    last_timestamp_secs = 0
    timestamp_objs = []
    values = []
    if timestamp_delim is None:
        timestamp_delim = "@"
    for i, timestamped_value in enumerate(vector):
        try:
            value_text = ""
            timestamp_text = ""
            if isinstance(timestamped_value, (list, tuple)):
                (timestamp_text, value_text) = timestamped_value
            else:
                data = timestamped_value.split(timestamp_delim, 1)
                timestamp_text = data[1]
                value_text = data[0]
            ## TODO: timestamp_objs.append(timestamp_text_to_secs(timestamp_text))
            ## HACK: don't convert into seconds if already integral
            timestamp_secs = timestamp_text
            if (not isinstance(timestamp_secs, int)):
                timestamp_secs = timestamp_text_to_secs(timestamp_text)
            if max_timestamp_secs and (timestamp_secs > max_timestamp_secs):
                debug.trace_fmt(4, "Ignoring sequence remainder ({n} items)) as max time reached ({mt})",
                                n=(len(vector) - i), mt=max_timestamp_secs)
                break
            timestamp_objs.append(timestamp_secs)
            values.append(float(value_text) if numeric else value_text)
            debug.assertion(last_timestamp_secs <= timestamp_secs) 
            last_timestamp_secs = timestamp_secs
        except ValueError:
            debug.trace_fmt(2, "Error: Unable to extract timestamp/value pair from {tv}", tv=timestamped_value)
    debug.trace_fmtd(5, "extract_timestamped_values() => {ts}, {vals}", ts=timestamp_objs, vals=values)
    return (timestamp_objs, values)

#-------------------------------------------------------------------------------

DEFAULT_LAG_MAX = 100
LAG_MAX = system.getenv_int("LAG_MAX", DEFAULT_LAG_MAX)
USE_CONVOLUTION = system.getenv_bool("USE_CONVOLUTION", False)

def ccf(v1, v2, lag_max=LAG_MAX, convolution=None):
    """Compute cross correlation for vectors V1 and V2 with normalization to match R's ccf function: the result will be up to |v1| + |v2|), depnding on lag size.
    Notes:
    - Use LAG_MAX parameter or env. var to specify maximum lag (defaults to {dlm}).
    - With CONVOLUTION enabled, the signal processing style correlation is used; otherwise, it is analogous to Pearson's r.
    - The scipy.signal.correlate method is 'direct' (i.e., for correlation).
    - Based on https://stackoverflow.com/questions/53959879/how-do-i-get-rs-ccf-in-python.
    """.format(dlm=DEFAULT_LAG_MAX)
    debug.trace(7, f"ccf({v1}, {v2}, [lag={lag_max}, use_conv={convolution})")
    if convolution is None:
        convolution = USE_CONVOLUTION

    # Perform the type of cross correlation desired
    if convolution:
        result = ss.correlate(v1, v2, method='direct')
    else:
        result = ss.correlate(v1 - np.mean(v1), v2 - np.mean(v2), method='direct')
        std_std_len = np.std(v1) * np.std(v2) * len(v2)
        if (std_std_len > 0):
            result /= std_std_len

    # Prune entries if larger than the lag
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    # Do sanity checks and return
    pruned_result = result[lo:hi]
    pruned_len = len(pruned_result)
    debug.assertion(0 < pruned_len <= len(result))
    ## TODO: debug.assertion(pruned_len >= min(lag_max, len(v1), len(v2)))
    debug.trace(5, f"ccf({v1}, {v2}) => {pruned_result}")
    
    return pruned_result

#-------------------------------------------------------------------------------

class CrossCorrelation(object):
    """Class for computing cross correlation of time-stamped data
    Note: the sequences were originally extended by default to have matching endpoints;
    however, relativized timestamps generally is preferred."""

    def __init__(self, vector1, vector2, *args, normalize=None, use_timestamps=False, extend_vectors=None, relativize_timestamps=None, time_increment=None, timestamp_delim=None, **kwargs):
        """Class constructor"""
        debug.trace_fmtd(5, "CrossCorrelation.__init__({v1}, {v2}, [{a}], normal={n} use_ts={uts}, ts_incr={tsi} delim={tsd}): keywords={kw}; self={s}",
                         v1=vector1, v2=vector2, n=normalize, uts=use_timestamps, tsi=time_increment, tsd=timestamp_delim,
                         a=",".join(args), kw=kwargs, s=self)
        # TODO: streamline handling of defaults (e.g., 'FUBAR = system.getenv("FUBAR", ...); ... def __init__(..., fubar=None, ...) ... if fubar is None: fubar = FUBAR; self.fubar = fubar' => 'self.x = system.get_kwarg_or_env(kwarg, 'X', ...)')
        self.vector1 = vector1
        self.vector2 = vector2
        self.normalize = normalize
        self.use_timestamps = use_timestamps
        if normalize is None:
            normalize = NORMALIZE_CCF
        if extend_vectors is None:
            extend_vectors = EXTEND_SEQUENCES
        self.extend_vectors = extend_vectors
        if time_increment is None:
            time_increment = ONE_DAY
        self.time_increment = time_increment
        if relativize_timestamps is None:
            relativize_timestamps = RELATIVIZE_TIMESTAMPS
        self.relativize_timestamps = relativize_timestamps
        debug.assertion(not (self.extend_vectors and self.relativize_timestamps))
        self.time_increment = time_increment
        self.timestamp_delim = timestamp_delim
        if self.timestamp_delim is None:
            self.timestamp_delim = "@"
        super(CrossCorrelation, self).__init__(*args, **kwargs)

    def trace_timestamped_vector(self, level, label, timestamps, vector):
        """Traces out zipped TIMESTAMPS and VECTOR with LABEL if at trace LEVEL
        Note: If timestamp delimiter defined, then format is value<delim>time"""
        delta_diff = np.array([0])
        if (len(timestamps) > 1):
            delta_diff = np.array([t2 - t1 for t1 in timestamps[0:-1] for t2 in timestamps[1:]])
            debug.trace_fmt(level, "{lbl} sequence for len={len} avg_time_delta={avg} stdev={stdev}:",
                            lbl=label, len=len(timestamps), avg=delta_diff.mean(), stdev=delta_diff.std())
            zlist = list(zip(timestamps, vector))
            if self.timestamp_delim:
                v = [f"{value}{self.timestamp_delim}{time}" for (time, value) in zlist]
                debug.trace_fmt(level, "\t{pp}", pp=", ".join(v))
            else:
                debug.trace_fmt(level, f"\t{zlist}")
        return

    def align_timestamped_vectors(self):
        """Re-align the two timestamped vectors and also expand so that the values represent uniform periods.
        Notes:
        - If self.extend_vectors, then each vector is enlarged to have matching endpoints.
        - If self.relativize_timestamps, then timestamps left as is (TODO, convert to 0-based)."""
        debug.trace(5, f"CrossCorrelation.align_timestamped_vectors(); self={self}")
        (timestamps1, self.vector1) = extract_timestamped_values(self.vector1, numeric=True, timestamp_delim=self.timestamp_delim)
        (timestamps2, self.vector2) = extract_timestamped_values(self.vector2, numeric=True, timestamp_delim=self.timestamp_delim)

        # Make sure the vectors span from combined min to combined max
        # Note: w/ relativize, this just resolves gaps within the time series (n.b., the timestamps are later ignored in the compute method, so this effectively makes both sets of timestamps start at zero).
        min1 = np.min(timestamps1)
        max1 = np.max(timestamps1)
        min2 = np.min(timestamps2)
        max2 = np.max(timestamps2)
        if self.extend_vectors:
            new_min = min(min1, min2)
            new_max = max(max1, max2)
            self.vector1 = self.rescale_time_series(new_min, new_max, timestamps1, self.vector1)
            self.vector2 = self.rescale_time_series(new_min, new_max, timestamps2, self.vector2)
        elif self.relativize_timestamps:
            self.vector1 = self.rescale_time_series(min1, max1, timestamps1, self.vector1)
            self.vector2 = self.rescale_time_series(min2, max2, timestamps2, self.vector2)
        else:
            system.print_stderr("Warning: vectors not expanded to fill timestamp gaps")
        self.trace_timestamped_vector(5, "v1", timestamps1, self.vector1)
        self.trace_timestamped_vector(5, "v2", timestamps2, self.vector2)
        return

    def rescale_time_series(self, min_time, max_time, timestamps, vector):
        """Extend VECTOR with TIMESTAMPS to ensure values from MIN_TIME to MAX_TIME and also expand so uniformly filled according to self.time_increments"""
        # Note: This inserts values in gaps based on the last value encountered
        # TODO: Allow for averaging of gap end-point values or simply using 0.0
        new_vector = []
        current_time = min_time
        pos = 0
        last_value = 0
        ## TODO: last_timestamp = 0
        next_time = 0
        if timestamps:
            ## TODO: last_timestamp = timestamps[-1]
            next_time = timestamps[0]
        debug.assertion(len(timestamps) == len(vector))
        while (current_time <= max_time):
            ## TODO: debug.assertion(isinstance(next_time, int))
            # Fill in next gap with multiple copies of last value
            while (current_time < next_time):
                new_vector.append(last_value)
                current_time += self.time_increment
            # Copy current value from time-stamped data
            if (pos < len(vector)):
                new_vector.append(vector[pos])
                last_value = vector[pos]
                pos += 1
                next_time = (timestamps[pos] if (pos < len(timestamps)) else max_time)
            current_time += self.time_increment
        ## TODO: debug.assertion(new_vector[:-1] >= last_timestamp)
        debug.trace_fmtd(5, "rescale_time_series({mn}, {mx}, {ts}, {v}) => {nv}",
                         mn=min_time, mx=max_time, ts=timestamps,
                         v=timestamps, nv=new_vector)
        return new_vector

    def compute(self):
        """Compute the cross correlation with optional timestamp expansion
        Note: Returns [0] if invalid input"""
        # TODO: Return [] on error and adjust processing downstream accordingly.
        debug.trace_fmtd(5, "compute(); self={s}", s=self)
        correlations = np.array([0])
        try:
            if self.use_timestamps:
                self.align_timestamped_vectors()
            else:
                self.vector1 = [safe_float(v) for v in self.vector1]
                self.vector2 = [safe_float(v) for v in self.vector2]            
            debug.trace_fmtd(7, "v1={v1}\nv2={v2}", v1=self.vector1, v2=self.vector2)
            if self.vector1 and self.vector2:
                # TODO: put normalize in ccf (and reconcile w/ USE_CONVOLUTION)
                if self.normalize:
                    correlations = ccf(self.vector1, self.vector2)
                else:
                    correlations = np.correlate(self.vector1, self.vector2)
            else:
                debug.trace(5, "Skipping cross correlation due to empty vector")
        except(ValueError):
            debug.trace_fmtd(4, "Exception in CrossCorrelation.compute: {exc}",
                             exc=sys.exc_info())
        debug.trace_fmtd(6, "compute() => {c}\n\tv1={v1}\n\tv2={v2}", c=correlations,
                         v1=self.vector1, v2=self.vector2)
        return correlations

#--------------------------------------------------------------------------------

class Script(Main):
    """Input processing class"""
    col1 = 1
    col2 = 2
    vector1 = []
    vector2 = []
    normalize = False
    use_timestamps = False
    compute_stats = False
    # TODO: delim = ','
    # TODO: time_increment = N
    # TODO: timestamp_delim = S

    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.col1 = self.get_parsed_option(COL1, self.col1)
        self.col2 = self.get_parsed_option(COL2, self.col2)
        ## TODO: self.var = self.get_parsed_option(VAR, self.var)
        self.normalize = self.get_parsed_option(NORMALIZE, self.normalize)
        self.use_timestamps = self.get_parsed_option(TIMESTAMP, self.use_timestamps)
        self.compute_stats = self.get_parsed_option(COMPUTE_STATS, self.compute_stats)
        ## TODO: self.filename = self.get_parsed_argument(FILENAME)
        tpo.trace_object(self, label="Script instance")
        return

    def process_line(self, line):
        """Processes current line from input"""
        debug.trace_fmtd(5, "Script.process_line(): self={s}", s=self)

        # Ignore comments
        line = line.strip()
        if (line.startswith("#") or line.startswith(";")):
            debug.trace_fmtd(4, "Ignoring comment line {n}: {l}",
                             n=(1 + self.line_num), l=line)
            return

        # Split into values and extract column values for each vector
        data = line.split(DELIM)
        if (len(data) < self.col2):
            debug.trace_fmtd(3, "Insufficient data at line {n}: {l}",
                             n=(1 + self.line_num), l=line)
        else:
            self.vector1.append(data[self.col1 - 1])
            self.vector2.append(data[self.col2 - 1])
        return

    def wrap_up(self):
        """Process the data that had been collected"""
        # TODO: add time_increment=self.time_increment, timestamp_delim=self.timestamp_delim)
        cross_corr = CrossCorrelation(self.vector1, self.vector2,
                                      normalize=self.normalize,
                                      use_timestamps=self.use_timestamps)
        correlations = cross_corr.compute()
        # c = cc
        print("cross-correlation: {cc}".format(cc=correlations))
        if self.compute_stats:
            print("correlation statistics:")
            print("avg: {v}".format(v=np.mean(correlations)))
            print("stdev: {v}".format(v=np.std(correlations)))
            print("min: {v}".format(v=np.min(correlations)))
            print("max: {v}".format(v=np.max(correlations)))
        return

#-------------------------------------------------------------------------------
    
if __name__ == '__main__':
    # TODO: copy trace_current_context into debug
    tpo.trace_current_context(level=tpo.QUITE_DETAILED)
    app = Script(
        description=__doc__,
        ## TODO: FIELD_DELIM, TIME_INCREMENT, TIMESTAMP_DELIM
        boolean_options=[
            (NORMALIZE, "Normalize the cross-correlation to match usage in statistics (e.g., R ccf)"),
            (TIMESTAMP, "Input values have timestamp (e.g., 13@2/21/2019)")],
        int_options=[COL1, COL2],
        ## TODO: positional_options=[FILENAME]
        positional_options=[])
    app.run()
