#! /usr/bin/env python
# 
# Extracts temporal-based features from patient claims data. This is initially
# in support of deriving features based on time-series, such as via cross
# correlation (see compute_cross_correlation.py). In addition, it was used
# used to support new features, such as sequence of ICD codes with associated
# timestamp, or ngrams based on ICD codes. This has been extended to handle
# arbitrary sequence data files, each using the same key and with a time-stamped
# value:
#      key<TAB>date<TAB>value
# For example,
#      ID,  date,       code
#      123, 2019-12-07, H2036
#      456, 2019-02-20, J0585
#      789, 2020-01-27, H2036
#      987, 2020-06-11, J0585
#
# Sample input (TODO: update):
# - Data from diagnostics (e.g., patient_id, service_date, diag1).
# - Data from prescription (e.g., patient_id, service_date, ndc).
# - Data from last_data.csv (e.g., patient_id & index_date).
# - Frequency counts from the above or from similar data.
#
# Notes:
# - The binary feature support is not fully functional.
# - This is in support of a small feasibility study to see whether temporal anaysis
#   of the sequences would lead to improvement in classification, such as for
#   drug-resistant epilepsy (DRE).
# - A full-scale solution would require use of a map-reduce framework, such as
#   Hadoop or Spark.
# - See following paper for details
#     An, et al. (2018), "Predicting drug-resistant epilepsy--A machine learning approach based
#     on administrative claims data", Epilepsy & Behavior, Volume 89, December 2018, Pages 118-125.
#
#-------------------------------------------------------------------------------
# TODO:
# - Show anonymized or mock sample data.
# - Include option for TODO: BLAH [???].
# - Change patient_id to key.
# - Use K for N when cluster numbers are involved (e.g., a la "k-means" mnemonic)
#

"""Extract features from patient claims data, such as time-series analysis
of diagnoses made, prescriptons filled, and procedures performed"""

# Standard packages
from collections import defaultdict
import csv
import functools
import random
import re
import sys

# Installed packages
import numpy as np
import pandas as pd

# Local packages
import debug
import glue_helpers as gh
from new_compute_cross_correlation import CrossCorrelation, timestamp_text_to_secs, extract_timestamped_values
from main import Main
import system
from system import round_as_str as rnd
import tpo_common as tpo

# Notes:
# - The constants for switches use dashes (e.g., DEBUG_MODE = "debug-mode"): underscore derived via main.py.
# - The old-style analysis was just for diagnoses, prescriptions, and procedures. Now arbitray sequence data
#   is supported (see SEQUENCE_DATA_FILES usage).
ENUMERATED_CODES = "enumerated-codes"
MAX_CODES = "max-codes"
DIAGNOSTICS_FILENAME = "diagnostics-filename"
PRESCRIPTIONS_FILENAME = "prescriptions-filename"
PROCEDURES_FILENAME = "procedures-filename"
VALID_INDEX_FILENAME = "valid-index-filename"
# 
TOP_DIAGNOSTICS = "top-diagnostics"
TOP_PRESCRIPTIONS = "top-prescriptions"
TOP_PROCEDURES = "top-procedures"
## OLD: PATIENT_FIELD = "patient_id"
## TODO: * make KEY_FIELD a command line option (also, assume first as default)!
KEY_FIELD = system.getenv_text("KEY_FIELD", "key")
CLASS_FIELD = system.getenv_text("CLASS_FIELD", "class")
TIME_INCREMENT = "time-increment"
OUTPUT_DELIM = "output-delimiter"
INPUT_DELIM = "input-delimiter"
INPUT_SORTED = "input-sorted"
CLUSTER_PATIENTS = "cluster-patients"
STARTING_CASE = "starting-case"
NUM_CASES = "num-cases"
# Note: old style diag/presc/proc
DIAGNOSTICS_PREFIX = "diag"
PRESCRIPTIONS_PREFIX = "presc"
PROCEDURES_PREFIX = "proc"
# 
SEQUENCE_DATA_FILENAMES = "sequence-data-filenames"
SEQUENCE_LABELS = "sequence-labels"
#
ONE_DAY = (60 * 60 * 24)                # number of seconds in a day (86,400)
ONE_MONTH = (ONE_DAY * (365.25/12))     # average number of seconds in a month (2,629,800)
#
CLASS_FILENAME = "class-filename"       # class for each case (added to csv for for use in machine learning)
AUX_DATA_FILENAME = "aux-data-filename" # additional data for output features

# Environment options
# NOTE: For the most part, this is used for debugging or for adhoc experimentation.
# TODO: Figure out Emacs indentation problem w/ getenv_xyz description.
# FUBAR = system.getenv_text("FUBAR", "F00Bar", "An option fouled up beyond all recognition")
#
REQUIRE_SEQUENCE_INTERSECTION = system.getenv_bool("REQUIRE_SEQUENCE_INTERSECTION", False,
                                                   "Take intersection of patients (e.g, procedures, diagnostics, etc.)")
# Also see EXTEND_SEQUENCES (and RELATIVIZE_TIMESTAMPS) in new_compute_cross_correlation.py.
USE_OLD_EXTRACTION = system.getenv_bool("USE_OLD_EXTRACTION", False,
                                        "Use old extraction technique (redundant processing)")
PRUNE_SEQUENCES = system.getenv_bool("PRUNE_SEQUENCES", False,
                                     "Prunes the sequences (e.g., no entries past index date)")
SKIP_SEQUENCE_PRUNING = (not PRUNE_SEQUENCES)
SORT_SEQUENCES = system.getenv_bool("SORT_SEQUENCES", False,
                                    "Sort input sequences (e.g., timestamped procedures)")
USE_SORTED_ENUMERATION = system.getenv_bool("USE_SORTED_ENUMERATION", False,
                                            "Sort enumeration codes to avoid arbitrary integers")
ALREADY_ENUMERATED = system.getenv_bool("ALREADY_ENUMERATED", False,
                                        "Whether enumeration codes already sorted")
INCLUDE_SELF_CORRELATIONS = system.getenv_bool("INCLUDE_SELF_CORRELATIONS", False,
                                               "Include features based on correlations against same patient (e.g., diagnostics vs. procedures)")
# debugging options
RANDOM_CENTROIDS = system.getenv_bool("RANDOM_CENTROIDS", False,
                                      "Select cluster prototype at random")
SEED = system.getenv_int("SEED", None, 
                         "Seed for random-number generated (if non-null)")
NUM_CLUSTERS = system.getenv_int("NUM_CLUSTERS", 10,
                                 "Number of random clusters")
JUST_CENTROIDS = system.getenv_bool("JUST_CENTROIDS", False,
                                    "Only compare the centroids against themselves")
USE_OLD_STYLE_INPUT = system.getenv_bool("OLD_STYLE_INPUT", False,
                                     "Assume fixed sequences: diagnostic/prescriptions/procedures [for regression testing]")
USE_NEW_STYLE_INPUT = (not USE_OLD_STYLE_INPUT)

# Sanity check(s)
debug.assertion(sys.version_info.major >= 3)

#--------------------------------------------------------------------------------

def create_lookup_set(filename, max_entries=None):
    """Create set for performing lookups from FILENAME, optionally with up to MAX_ENTRIES
    Note: Format of the file in <KEY>[<TAB><VALUE>], with optional value ignored (e.g., frequency)"""
    debug.trace_fmt(5, "create_lookup_set({f}, [{m}])", f=filename, m=max_entries)
    lookup_set = set()
    line_num = 0
    for line in open(filename):
        line = line.strip()
        line_num += 1
        debug.trace_fmt(6, "[{f}] L{n}: {l}", f=filename, n=line_num, l=line)
        if (max_entries and len(lookup_set) >= max_entries):
            debug.trace_fmt(5, "Stopping as max entries reached: {m}", m=max_entries)
            break
        entry = re.sub(r"^([^\t]+).*", r"\1", line)
        debug.assertion(entry not in lookup_set)
        lookup_set.add(entry)
    debug.trace_fmt(6, "create_lookup_set() => s; len(s)={l}", l=len(lookup_set))
    debug.trace_fmt(7, "\ts:{{{s}}}", s=lookup_set)
    debug.assertion(len(lookup_set) > 0)
    return lookup_set


def nth(list_or_tuple, n):
    """Returns the n-th element of LIST_OR_TUPLE or None if not applicable (e.g., empty list)
    note: Used to define first, second, etc, so the docstring for the latter might be misleading"""
    ## EX: nth(("fido", 13), 2) => 13
    ## EX: nth([3, 2, 1], 3) => 1
    ## EX: nth([], 1) => None
    value = None
    try:
        value = list_or_tuple[n]
    except(TypeError):
        debug.trace_fmtd(6, "Exception in nth: {exc}", exc=sys.exc_info())
    return value


## TODO: define common cases with specific docstring
## def first(list_or_tuple):
##     """Return first item in LIST_OR_TUPLE"""
##     return (nth(list_or_tuple, 1))
#
## TODO: for i, fn in enumerate(["second", ..., "fifth"): eval(fn + " lambda t: nth(t, " + str(i))
first = lambda t: nth(t, 0)
second = lambda t: nth(t, 1)
third = lambda t: nth(t, 2)


def tuples_are_sorted(tuple_list, key_fn=first):
    """Verify that TUPLE_LIST is already sorted with respect to KEY (function first by default)"""
    # EX: tuples_are_sorted([(2, "dos"), (3, "tres"), (1, "uno")], key_fn=second)
    is_sorted = all([(t1 is None) or (key_fn(t1) <= key_fn(t2))
                     for t1 in tuple_list for t2 in tuple_list[2:]])
    return (is_sorted)


ELIDED_MAX_NUM = 25
#
def elide_sequence(sequence, max_num=ELIDED_MAX_NUM):
    """Return string version of sequence elided if over MAX_NUM ({emn}) elements)""".format(emn=ELIDED_MAX_NUM)
    # EX: elide_sequence([1, 2, 3, 4, 5], 3) => "[1, 2, 3, ...]"
    output_seq = sequence
    if (len(sequence) > max_num):
        output_seq = output_seq[:max_num] + [", ..."]
    seq_spec = system.to_string(output_seq)
    return seq_spec


def lookup_df_row(data_frame, lookup_value, lookup_field=KEY_FIELD):
    """Return first row of DATA_FRAME with LOOKUP_FIELD having LOOKUP_VALUE"""
    ## TODO: rework in terms of Pandas primitive
    row = None
    matches = [row for _index, row in data_frame.iterrows() if (row[lookup_field] == lookup_value)]
    if matches:
        row = matches[0]
    debug.trace(7, f"lookup_df_row(_, {lookup_field}, {lookup_value}) => {row}")
    return row



#--------------------------------------------------------------------------------

class SequenceCollector(object):
    """Class for collecting sequence of timestamped data for patients"""
    
    def __init__(self, *args, csv_delimiter=None, **kwargs):
        """Class constructor"""
        debug.trace_fmtd(5, "SequenceCollector.__init__({a}, [csv_delimiter={cd}): keywords={kw}; self={s}",
                         a=",".join(args), cd=csv_delimiter, kw=kwargs, s=self)
        self.patient_data = defaultdict(list)
        if csv_delimiter is None:
            csv_delimiter = ""
        self.csv_delimiter = csv_delimiter
        super(SequenceCollector, self).__init__(*args, **kwargs)
        return

    def read_patient_data_ws(self, filename):
        """Read patient data file with fields ID, TIMESTAMP, and CODE, saving it in a hash keyed off
        of ID with list of (TIME_VALUE, CODE) pairs.
        Note: This uses whitespace (ws) as a delimiter; use read_patient_data_csv for comma-separated value input."""
        debug.trace_fmt(4, "Warning: using obsolete method {old}: use {new} instead.",
                        old="read_patient_data_ws", new="read_patient_data")
        line_num = 0
        num_cases = 0
        for line in open(filename):
            line = line.strip()
            line_num += 1
            debug.trace_fmt(7, "[{f}] L{n}: {l}", f=filename, n=line_num, l=line)

            # Skip header (field labels or separator line with dashes)
            if (line.startswith(KEY_FIELD) or line.startswith("---")):
                debug.trace_fmt(7, "Ignoring header at line {n}", n=line_num)
            # Otherwise split into fields and save in hash 
            else:
                data = line.split()
                if (len(data) != 3):
                    debug.trace_fmt(3, "Warning: {cols} field(s) found, 3 expected at {fn}:{n}): [l]",
                                    cols=len(data), fn=filename, n=line_num, l=line)
                else:
                    num_cases += 1
                    (patient_id, time, code) = data
                    ## OLD: self.patient_data[patient_id].append("{c}@{t}".format(c=code, t=time))
                    self.patient_data[patient_id].append((timestamp_text_to_secs(time), code))
                    ## TODO: self.patient_data[patient_id].append((time, code))
        debug.trace_fmt(5, "Read {nd} distinct cases of {nt} total",
                        nd=len(self.patient_data), nt=num_cases)

        # Ensure sorted
        # TODO: Use already-sorted test to avoid sorting overhead
        if SORT_SEQUENCES:
            for patient_id in self.patient_data:
                self.patient_data[patient_id] = sorted(self.patient_data[patient_id],
                                                       key=first)
        else:
            debug.assertion(tuples_are_sorted(self.patient_data[patient_id], key_fn=first))
        return

    def read_patient_data(self, filename):
        """Read patient data file with fields ID, TIMESTAMP, and CODE, saving it in a hash keyed off
        of ID with list of (TIME_VALUE, CODE) pairs"""
        # TODO: rename at read_patient_data_csv???
        # Note: Support for old format (whitespace delimited)
        if (self.csv_delimiter == ""):
            self.read_patient_data_ws(filename)
            return

        fh = open(filename, "r")
        csv_reader = csv.reader(iter(fh.readlines()), delimiter=self.csv_delimiter, quotechar='"')
        num_cases = 0
        headers = []
        for i, row in enumerate(csv_reader):
            debug.trace_fmt(6, "[{f}] L{n}: {r}", f=filename, n=(i + 1), r=row)
            if i == 0:
                headers = row
                debug.assertion(KEY_FIELD in headers)
                continue

            if (len(row) != 3):
                debug.trace_fmt(3, "Warning: {cols} fields found, 3 expected at {fn}:{n}): {r}",
                                cols=len(row), fn=filename, n=(i + 1), r=row)
            else:
                (patient_id, time, code) = row
                if code.strip():
                    num_cases += 1
                    self.patient_data[patient_id].append((timestamp_text_to_secs(time), code))
                else:
                    debug.trace_fmt(4, "Warning: ignoring blank code at line {fn}:{n}: '{c}'",
                                    fn=filename, n=(i + 1), c=code)
        debug.trace_fmt(5, "Read {nd} distinct cases of {nt} total",
                        nd=len(self.patient_data), nt=num_cases)

        # Ensure sorted
        # TODO: Use already-sorted test to avoid sorting overhead
        for patient_id in self.patient_data:
            self.patient_data[patient_id] = sorted(self.patient_data[patient_id],
                                                   key=(lambda tuple: tuple[0]))
        return

    def get_ids(self):
        """Return list of IDs in hash"""
        return list(self.patient_data.keys())

    def get_data(self, key):
        """Return list of tuples for KEY sorted by timestamp"""
        return self.patient_data[key]

#................................................................................    

class TemporalFeatureExtractor(object):
    """Class for extracting features based on temporal sequences"""

    def __init__(self, valid_index_date, *args, use_enumerated_codes=False, time_increment=ONE_MONTH, use_sorted_enumeration=None, already_enumerated=None, **kwargs):
        """Class constructor"""
        debug.trace_fmtd(5, "TemporalFeatureExtractor.__init__(_, uec={uec}, ti={ti}, use={use}, ae={ae}, [{a}]): keywords={kw}; self={s}",
                         uec=use_enumerated_codes, ti=time_increment, use=use_sorted_enumeration, ae=already_enumerated,
                         a=",".join(args), kw=kwargs, s=self)
        debug.trace_fmtd(6, "\tvalid_index_date={vid}", vid=valid_index_date)
        self.patient_data = defaultdict(list)
        self.valid_index_date = valid_index_date
        self.valid_index_secs = defaultdict(int)
        self.use_enumerated_codes = use_enumerated_codes
        self.time_increment = time_increment
        if use_sorted_enumeration is None:
            use_sorted_enumeration = USE_SORTED_ENUMERATION
        self.use_sorted_enumeration = use_sorted_enumeration
        if already_enumerated is None:
            already_enumerated = ALREADY_ENUMERATED
        self.already_enumerated = already_enumerated
        ## TODO: super(TemporalFeatureExtractor, self).__init__(*args, **kwargs)
        ## NOTE: it's so moronic that object contructor not take argument (e.g., stupid maintenance issue when changing base class)
        super(TemporalFeatureExtractor, self).__init__()
        return

    def extract_temporal_sequences(self, key, label, timestamped_sequence):
        """Extract the timestamped sequence of codes (e.g., ICD), resolving textual timestamps into seconds, and making sure the entries are not past the index date for the patient"""
        ## OLD: (implicitly) expanded sequences based on time deltas
        debug.trace_fmt(4, "extract_temporal_sequences({k}, {l}, _)", k=key, l=label)
        debug.trace_fmt(5, "\tlen={l} seq={ts}", l=len(timestamped_sequence), ts=timestamped_sequence)
        # Note: The sequences might already pruned by index date, so SKIP_SEQUENCE_PRUNING is used
        # to streamline the processing.
        if SKIP_SEQUENCE_PRUNING:
            debug.trace_fmt(5, "Returning input sequence of len {l} as is (e.g., not pruned): {ts}",
                            l=len(timestamped_sequence), ts=elide_sequence(timestamped_sequence))
            return timestamped_sequence

        # Make sure sequences don't extend past valid index dates: see (An et al. 2018) paper.
        # Note: All timestamps used if no index file; and, none used unless index date explicitly given for patient.
        valid_index_date = None
        valid_index_timestamp_secs = system.maxint()
        if self.valid_index_date:
            valid_index_date = self.valid_index_date.get(key, "")
            if (not valid_index_date):
                debug.trace_fmt(2, "Warning: unable to resolve index date for key '{k}'", k=key)
            debug.assertion(valid_index_date != "")
            valid_index_timestamp_secs = (timestamp_text_to_secs(valid_index_date) if valid_index_date else 0)
        self.valid_index_secs[key] = valid_index_timestamp_secs
        if not USE_OLD_EXTRACTION:
            pruned_timestamped_sequence = [(t, c) for (t, c) in timestamped_sequence if (t <= valid_index_timestamp_secs)]
            debug.trace_fmt(5, "Returning pruned sequence of len {l} (was len {old_l): {ps}",
                            l=len(pruned_timestamped_sequence), old_l=len(timestamped_sequence),
                            ps=elide_sequence(pruned_timestamped_sequence))
            return pruned_timestamped_sequence

        # OLD: Separate the timestamps from the data, and filter dates past index point.
        # Also, trace out statistics (e.g., time delta averages).
        (timestamps, vector) = extract_timestamped_values(timestamped_sequence, max_timestamp_secs=valid_index_timestamp_secs)

        # Convert result back into single list.
        extracted_sequence = list(zip(timestamps, vector))
        debug.trace_fmt(5, "Returning extracted sequence of len {l} via old approach: {es}",
                        l=len(extracted_sequence), ps=elide_sequence(extracted_sequence))
        return extracted_sequence

    def derive_cross_correlation(self, subseq1, subseq2, bad_code=-123):
        """Compute the cross correlation for SUBSEQ1 and SUBSEQ2, retuning BAD_CODE if an error (e.g., empty sequence)"""
        ## Note: control of sequence expansion is handled by CrossCorrelation class (e.g., using time_increment)
        cc_result = bad_code
        if (subseq1 and subseq2):
            cc = CrossCorrelation(subseq1, subseq2, normalize=True, use_timestamps=True, time_increment=self.time_increment)
            cc_result = cc.compute().mean()
        return cc_result

    def create_enumeration(self, timestamped_sequence):
        """Return a hash with the enumeration codes for each element of TIMESTAMPED_SEQUENCE
        Note: The code sequences are assigned arbitrary integers unless self.use_sorted_enumeration set, which is intended to produce better (cross) correlations"""
        # TODO: incorporate domain knowledge on how the codes are related; allow for real-valued codes to allow for modelling distinctions in code distances
        # TODO: handle already_enumerated case elsewhere
        if self.use_sorted_enumeration:
            timestamped_sequence = sorted(timestamped_sequence, key=second)
        if self.already_enumerated:
            enum_codes = {c: c for (_t, c) in timestamped_sequence}
        else:
            enum_codes = {c: i for (i, (_t, c)) in enumerate(timestamped_sequence)}
        return enum_codes

    def extract_cross_correlations(self, key, timestamped_sequence1, top_codes1, timestamped_sequence2, top_codes2, timestamped_sequence3=None, top_codes3=None):
        """Extract average cross correlations among TIMESTAMPED_SEQUENCE1 and TIMESTAMPED_SEQUENCE2, based on
        TOP_CODES1 and TOP_CODES2, resulting in a matrix (or vector if enumerated codes used)
        Note: assumes extract_temporal_sequences run to produce the sequences (and record timestamp secs valid for the KEY)"""
        # Notes:
        # - First sequence is currently diagnostics, the second prescriptions, and the third procedures.
        # - By default, converts each sequence of codes into multiple binary sequences of occurrences.
        #   -- Requires M cross-correlations per patient, yielding MxM matrix, with M << N (#Codes).
        #   -- Would yield N features (one per code).
        # TODO: Run cross correlations against same sequence (not just seq1 vs seq2)
        #   -- Would produce 3x avg correlations: s1 vs s1, s1 vs s2, s2 vs s2
        ## DEBUG: avg_correlations = np.zeros((len(top_codes1), len(top_codes2)), dtype=str)
        # TODO: rename as something like calc_cross_correlations -or- extract_pairwise_cross_correlations
        debug.trace_fmt(8, "extract_cross_correlations({k}, {ts1}, {tc1}, {ts2}, {tc2}, {ts3}, {tc3})", 
                        k=key, ts1=timestamped_sequence1, tc1=top_codes1, ts2=timestamped_sequence2, 
                        tc2=top_codes2, ts3=timestamped_sequence3, tc3=top_codes3)

        # HACK: If using enumerated codes, just produce one cross correlation
        if self.use_enumerated_codes:
            # NOTE: The integral code for a value is based on its last position in the sequence.
            # TODO: Incorporate domain knowledge to produce reasonable enumerations (not arbitrary).
            code_number1 = self.create_enumeration(timestamped_sequence1)
            subseq1 = [(t1, code_number1[c1]) for (t1, c1) in timestamped_sequence1]
            code_number2 = self.create_enumeration(timestamped_sequence2)
            subseq2 = [(t2, code_number2[c2]) for (t2, c2) in timestamped_sequence2]
            code_number3 = self.create_enumeration(timestamped_sequence3)
            subseq3 = [(t3, code_number3[c3]) for (t3, c3) in timestamped_sequence3]
            cc_s1_s1 = self.derive_cross_correlation(subseq1, subseq1)
            cc_s1_s2 = self.derive_cross_correlation(subseq1, subseq2)
            cc_s1_s3 = self.derive_cross_correlation(subseq1, subseq3)
            cc_s2_s2 = self.derive_cross_correlation(subseq2, subseq2)
            cc_s2_s3 = self.derive_cross_correlation(subseq2, subseq3)
            cc_s3_s3 = self.derive_cross_correlation(subseq3, subseq3)
            ## OLD: correlations = [cc_s1_s1.mean(), cc_s1_s2.mean(), cc_s1_s3.mean(), cc_s2_s2.mean(), cc_s2_s3.mean(), cc_s3_s3.mean()]
            correlations = [cc_s1_s1, cc_s1_s2, cc_s1_s3, cc_s2_s2, cc_s2_s3, cc_s3_s3]
            debug.trace_fmt(3, "Cross correlations (s1 vs s1, s1 vs s2, s1 vs s3, s2 vs s2, s2 vs s3):\n{c}", c=correlations)
            return correlations

        # Initialize matrix, including dummy spots for other codes
        # Make sure the entire array will be printed, using limited precision (6 => 3).
        # TODO: restore printoptions to defaults; account for top_codes3
        avg_correlations = np.zeros(((len(top_codes1) + 1), (len(top_codes2) + 1)))
        ## OLD: np.set_printoptions(threshold=np.nan)
        # note: threshold is max size of array before ellipsis used.
        np.set_printoptions(threshold=sys.maxsize)
        np.set_printoptions(precision=3)
        
        # Convert first sequence of codes to sequences of binary indicator for each top code
        # Note: dummy entry at end encodes non-code counts (e.g., [(t1, 1), (t2, 0), (t2 + 1, 5)])
        # TODO: account for timestamped_sequence3
        debug.assertion(timestamped_sequence3 is None)
        for i, (time1, code1) in enumerate(timestamped_sequence1):
            debug.assertion((time1 <= self.valid_index_secs[key]) or (i == (len(timestamped_sequence1) - 1)))
            if (code1 not in top_codes1):
                debug.trace_fmt(5, "Skipping non-top code 1 {c}", c=code1)
                continue
            subseq1 = [(t1, int(c1 == code1)) for (t1, c1) in timestamped_sequence1]
            debug.trace_fmt(3, "Subsequence1 {n}: {v}", n=(i + 1), v=subseq1)
            if not subseq1:
                continue
            subseq1.append((subseq1[-1][0] + 1, sum([c for (t, c) in subseq1])))
            row = list(top_codes1).index(code1)

            # Likewise convert second code sequence to binary indicator sequences for top code and cross correlate
            # TODO: Use helper function to encapsulate timestamp transformation (i.e., common to subseq1 and subseq2).
            for j, (time2, code2) in enumerate(timestamped_sequence2):
                debug.assertion((time2 <= self.valid_index_secs[key]) or (j == (len(timestamped_sequence2) - 1)))
                if (code2 not in top_codes2):
                    debug.trace_fmt(5, "Skipping non-top code 2 {c}", c=code2)
                    continue
                subseq2 = [(t2, int(c2 == code2)) for (t2, c2) in timestamped_sequence2]
                debug.trace_fmt(3, "Subsequence2 {n}: {v}", n=(j + 1), v=subseq2)
                if not subseq2:
                    continue
                subseq2.append((subseq1[-1][0] + 1, sum([c for (t, c) in subseq1])))

                # Perform cross correlation and record
                # TODO: factor in subseq3
                cc_s1_vs_s2 = self.derive_cross_correlation(subseq1, subseq2)
                ## OLD: correlations = cc.compute()
                ## OLD: debug.trace_fmt(3, "cross coorrelation {r} vs. {c}: {cc}", r=i, c=j, cc=correlations)
                debug.trace_fmt(3, "cross coorrelation {r} vs. {c}: {cc}", r=i, c=j, cc=cc_s1_vs_s2)
                col = list(top_codes2).index(code2)
                ## OLD: debug.assertion(avg_correlations[row, col] == 0)
                ## OLD: avg_correlations[row, col] = correlations.mean()
                ## OLD: avg_correlations[row, col] = correlations
                avg_correlations[row, col] = cc_s1_vs_s2
        debug.trace_fmt(3, "Average cross correlations (s1 vs. s2):\n{m}", m=avg_correlations)
        return avg_correlations

    def extract_paired_cross_correlations(self, key1, timestamped_sequence1, key2, timestamped_sequence2, sequence_label=None):
        """Extract cross correlations for SEQ1 vs SEQ2"""
        debug.trace_fmt(8, "extract_paired_cross_correlations({k1}, {ts1}, {k2}, {ts2}, {l})", 
                        k1=key1, ts1=timestamped_sequence1, k2=key2, ts2=timestamped_sequence2, l=sequence_label)

        # If using enumerated codes, just produce one cross correlation
        label = ((sequence_label + " ") if sequence_label else "")
        if self.use_enumerated_codes:
            # TODO: incorporate domain knowledge to produce reasonable enumerations (not arbitrary)
            # TODO: make sure sequences don't incorporate values past valid index date
            code_number1 = {c1: i for (i, (_t1, c1)) in enumerate(timestamped_sequence1)}
            subseq1 = [(t1, code_number1[c1]) for (t1, c1) in timestamped_sequence1]
            code_number2 = {c2: j for (j, (_t2, c2)) in enumerate(timestamped_sequence2)}
            subseq2 = [(t2, code_number2[c2]) for (t2, c2) in timestamped_sequence2]
            cc_s1_s2 = self.derive_cross_correlation(subseq1, subseq2)
            ## OLD: correlations = [cc_s1_s2.mean()]
            correlations = [cc_s1_s2]
            debug.trace_fmt(3, "Paired {l}cross correlations (s1 vs s2):\n{c}", c=correlations, l=label)
            return correlations

        system.print_stderr("Error: Binary indicators for enumeration codes not implemented")
        ## TODO: raise NotImplemetedError
        assert(False)
        return []
     
#-------------------------------------------------------------------------------

class Script(Main):
    """Adhoc script class to read datafiles required for processsing and to analyze timestamped series alone and pairwise"""
    max_codes = 100
    use_enumerated_codes = False
    diagnostics_filename = None
    prescriptions_filename = None
    procedures_filename = None
    top_diagnostics_filename = None
    top_prescriptions_filename = None
    top_procedures_filename = None
    valid_index_filename = None
    sequence_data_filenames = None
    sequence_labels = None
    time_increment = None
    sequence_labels = None
    # TODO: change default input delim to ","
    input_delim = " "
    output_delim = ","
    input_sorted = False
    use_centroid_patients = False
    num_clusters = 0
    cluster_patients = []
    starting_case = 1
    num_cases = -1
    ## TODO: num_casesnum_cases = sys.maxsize        # TODO: use package with max values
    class_filename = ""
    aux_data_filename = ""

    def setup(self):
        """Check results of command line processing"""
        debug.trace_fmtd(5, "Script.setup(): self={s}", s=self)
        self.max_codes = self.get_parsed_option(MAX_CODES, self.max_codes)
        self.use_enumerated_codes = self.get_parsed_option(ENUMERATED_CODES, self.use_enumerated_codes)
        self.time_increment = self.get_parsed_option(TIME_INCREMENT, self.time_increment)
        self.output_delim = self.get_parsed_option(OUTPUT_DELIM, self.output_delim)
        self.input_delim = self.get_parsed_option(INPUT_DELIM, self.input_delim)
        self.input_sorted = self.get_parsed_option(INPUT_SORTED, self.input_sorted)
        self.starting_case = self.get_parsed_option(STARTING_CASE, self.starting_case)
        self.num_cases = self.get_parsed_option(NUM_CASES, self.num_cases)
        if USE_OLD_STYLE_INPUT:
            self.top_diagnostics_filename = self.get_parsed_option(TOP_DIAGNOSTICS)
            self.top_prescriptions_filename = self.get_parsed_option(TOP_PRESCRIPTIONS)
            self.top_procedures_filename = self.get_parsed_option(TOP_PROCEDURES)
            self.valid_index_filename = self.get_parsed_argument(VALID_INDEX_FILENAME)
            self.diagnostics_filename = self.get_parsed_argument(DIAGNOSTICS_FILENAME)
            self.prescriptions_filename = self.get_parsed_argument(PRESCRIPTIONS_FILENAME)
            self.procedures_filename = self.get_parsed_argument(PROCEDURES_FILENAME)
        else:
            self.sequence_data_filenames = self.get_parsed_argument(SEQUENCE_DATA_FILENAMES).split()
            self.sequence_labels = self.get_parsed_option(SEQUENCE_LABELS)
            self.valid_index_filename = self.get_parsed_option(VALID_INDEX_FILENAME)
        cluster_patients = self.get_parsed_option(CLUSTER_PATIENTS, self.cluster_patients)
        if cluster_patients:
            self.cluster_patients = cluster_patients.split()
            self.num_clusters = len(self.cluster_patients)
            self.use_centroid_patients = True
        self.class_filename  = self.get_parsed_option(CLASS_FILENAME, self.class_filename)
        self.aux_data_filename  = self.get_parsed_option(AUX_DATA_FILENAME, self.aux_data_filename)
        tpo.trace_object(self, label="Script instance")

    def run_main_step(self):
        """Main processing step"""
        # TODO: place part of this code in temporal analysis class
        debug.trace_fmtd(5, "Script.run_main_step(): self={s}", s=self)
        if USE_OLD_STYLE_INPUT:
            return self.old_style_main_step()

        # Read optional valid index date--for limiting amount of sequence data for training
        patients_valid_index_date = None
        if self.valid_index_filename:
            patients_valid_index_date = system.read_lookup_table(self.valid_index_filename, 
                                                                 skip_header=True, delim=self.input_delim)
        patients_class_values = None
        if self.class_filename:
            patients_class_values = system.read_lookup_table(self.class_filename, delim=self.input_delim)
        if self.aux_data_filename:
            aux_data = pd.read_csv(self.aux_data_filename, delimiter=self.input_delim, dtype=str, error_bad_lines=False, comment="#")

        # If random centroids, generate prototypes from list of patients with index date
        if RANDOM_CENTROIDS:
            if SEED:
                random.seed(SEED)
            self.num_clusters = NUM_CLUSTERS
            debug.assertion(patients_valid_index_date)
            self.cluster_patients = sorted(list(patients_valid_index_date.keys()),
                                           key=lambda _tuple: random.random())[:self.num_clusters]
            self.use_centroid_patients = True
            debug.trace(4, f"random prototypes: {self.cluster_patients}")

        # Read each set of sequences, along with optional top-N file for each
        ## OLD
        ## num_sequences = len(self.sequence_data_filenames)
        ## sc_data = [None] * num_sequences
        ## top_data = [[]] * num_sequences
        common_ids = []
        if patients_valid_index_date:
            common_ids = list(patients_valid_index_date.keys())
        ## OLD: for i, filename in enumerate(self.sequence_data_filenames):
        # Note: the filenames can contain glob patterns, so resolved separately.
        sequence_files = [f for filespec in self.sequence_data_filenames for f in gh.get_matching_files(filespec)]
        #
        num_sequences = len(sequence_files)
        sc_data = [None] * num_sequences
        top_data = [[]] * num_sequences
        #
        for i, filename in enumerate(sequence_files):
            sc_data[i] = SequenceCollector(csv_delimiter=self.input_delim)
            sc_data[i].read_patient_data(filename)
            top_filename = "top-" + filename
            if system.file_exists(top_filename):
                top_data[i] = create_lookup_set(top_filename, self.max_codes)
            if REQUIRE_SEQUENCE_INTERSECTION:
                common_ids = system.intersection(common_ids, sc_data[i].get_ids())
        tfe = TemporalFeatureExtractor(patients_valid_index_date, use_enumerated_codes=self.use_enumerated_codes, time_increment=self.time_increment)
        if not common_ids:
            if REQUIRE_SEQUENCE_INTERSECTION:
                system.print_stderr("Error: empty set of common ID's (across ID's from sequences)")
                sys.exit()
            system.print_stderr("Warning: using union ID's from sequences for common ones: specify valid index file to override")
            ## OLD: common_ids = functools.reduce(system.union, [sc_data[i].get_ids() for i in range(num_sequences)])
            common_ids = list(functools.reduce(system.union, [sc_data[i].get_ids() for i in range(num_sequences)]))

        # Derive feature labels
        debug.assertion(self.use_enumerated_codes)
        feature_legend = ""
        if not self.sequence_labels:
            self.sequence_labels = [f"S{i + 1}" for i in range(num_sequences)]
            feature_legend = ("# sequence legend:\n" + 
                              "\n".join([f"# {l}: {f}" for (l, f) in zip(self.sequence_labels, sequence_files)]))
        non_temporal_feature_labels = []
        temporal_feature_labels = []
        if self.class_filename:
            non_temporal_feature_labels += [CLASS_FIELD]
        if self.aux_data_filename:
            debug.assertion(aux_data.columns[0] == KEY_FIELD)
            non_temporal_feature_labels += list(aux_data.columns[1:])
        if INCLUDE_SELF_CORRELATIONS:
            for i, label1 in enumerate(self.sequence_labels):
                for j, label2 in enumerate(self.sequence_labels):
                    if i <= j:
                        temporal_feature_labels.append(f"{label1}v{label2}")
        cluster_time_sequences = [[]] * self.num_clusters
        if self.use_centroid_patients:
            feature_legend += "# feature legend:\n# SivSi_cj: current Series_i case vs. Series_i prototype for cluster c\n"
            for c, cluster_patient_id in enumerate(self.cluster_patients):
                cluster_feature_labels = [f"{label}v{label}_c{c}" for i, label in enumerate(self.sequence_labels)]
                debug.assertion(not system.intersection(temporal_feature_labels, cluster_feature_labels))
                temporal_feature_labels += cluster_feature_labels

                cluster_time_sequences[c] = [None] * num_sequences
                for i, sc in enumerate(sc_data):
                    data_sequence = sc.get_data(cluster_patient_id)
                    label = self.sequence_labels[i]
                    cluster_time_sequences[c][i] = tfe.extract_temporal_sequences(cluster_patient_id, label, data_sequence)

        # Output the feature table with values calculated on the fly (e.g., cross correlation against
        # the various cached temporal sequences for prototypes).
        if feature_legend:
            sys.stdout.write(feature_legend)
        csv_writer = csv.writer(sys.stdout, delimiter=self.output_delim)
        csv_writer.writerow([KEY_FIELD] + non_temporal_feature_labels + temporal_feature_labels)
        start_pos = self.starting_case - 1
        num_cases = (self.num_cases if (self.num_cases >= 0) else max(0, (len(common_ids) - start_pos)))
        end_pos = (start_pos + num_cases)
        debug.trace_fmt(5, "Using case range [{s}, {e}]",
                        s=start_pos, e=(end_pos - 1))
        num_skipped = 0
        for patient_id in common_ids[start_pos:end_pos]:
            if (JUST_CENTROIDS and (patient_id not in self.cluster_patients)):
                debug.trace(4, f"Skipping non-centroid {patient_id}")
                num_skipped += 1
                continue
            debug.assertion(patient_id != KEY_FIELD)

            # Extract the sequences, resolving date strings into seconds
            # TODO: use cached value if prototype
            time_sequences = [None] * num_sequences
            for i, sc in enumerate(sc_data):
                time_sequences[i] = tfe.extract_temporal_sequences(patient_id, self.sequence_labels[i], sc.get_data(patient_id))

            # Add features for cross correlations against self (num = #Seq x #Seq-1)
            temporal_feature_values = []
            if INCLUDE_SELF_CORRELATIONS:
                for i, sc1 in enumerate(sc_data):
                    for j, sc2 in enumerate(sc_data):
                        if i <= j:
                            correlations = tfe.extract_cross_correlations(patient_id, 
                                                                          sc1, top_data[i], 
                                                                          sc2, top_data[j])
                            temporal_feature_values += [rnd(v) for v in correlations]

            # Add features for cross correlations against (prototypical) cluster patients (#Seq x K)
            for c, cluster_patient_id in enumerate(self.cluster_patients):
                debug.trace(5, f"{patient_id} vs. prototype {cluster_patient_id}")
                correlations = []
                for i, time_seq in enumerate(time_sequences):
                    correlations += tfe.extract_paired_cross_correlations(patient_id, time_seq, 
                                                                          cluster_patient_id, cluster_time_sequences[c][i], 
                                                                          sequence_label=self.sequence_labels[i])
                temporal_feature_values += [rnd(v) for v in correlations]

            # do sanity check on feature labels and values
            debug.trace_fmt(6, "temporal_feature_values: {tfv}", tfv=temporal_feature_values)
            debug.assertion(len(temporal_feature_values) == len(temporal_feature_labels))
            ## TEMP:
            if (len(temporal_feature_values) != len(temporal_feature_labels)):
                system.print_stderr("Unexpected condition: len(temporal_feature_values)={len1} len(temporal_feature_labels)={len2}: l1={l1} l2={l2}", len1=len(temporal_feature_values), len2=len(temporal_feature_labels), l1=temporal_feature_values, l2=temporal_feature_labels)

            # Print features (and trace out values and types)
            all_feature_values = [str(patient_id)] 
            if self.class_filename:
                all_feature_values += [patients_class_values[patient_id]]
            if self.aux_data_filename:
                row = lookup_df_row(aux_data, patient_id)
                debug.assertion(row[0] == patient_id)
                all_feature_values += list(row[1:])
            all_feature_values += temporal_feature_values
            debug.trace_fmt(7, "feature types: {ft}", ft=[type(v) for v in all_feature_values])
            debug.trace_array(6, all_feature_values, "feature values")
            csv_writer.writerow(all_feature_values)

        # Sanity checks
        if JUST_CENTROIDS:
            debug.assertion(num_skipped < num_cases)
            debug.trace(4, f"{num_skipped} of {num_cases} cases skipped")

        return None

    def old_style_main_step(self):
        """Run processing of procedure/dignostics/prescriptions
        Note: retained for regression testing purposes"""
        # Read patient "index date", the last date training can be done over (see paper)
        # note: ???
        patients_valid_index_date = system.read_lookup_table(self.valid_index_filename, 
                                                             skip_header=True, delim=self.input_delim)
        debug.assertion(len(patients_valid_index_date) > 0)

        # If random centroids, generate prototypes from list of patients with index date
        if RANDOM_CENTROIDS:
            if SEED:
                random.seed(SEED)
            ## TODO:
            ## num_patients = len(patients_valid_index_date)
            ## self.num_clusters = int(round(random.random() * num_patients))
            self.num_clusters = NUM_CLUSTERS
            self.cluster_patients = sorted(list(patients_valid_index_date.keys()),
                                           key=lambda _tuple: random.random())[:self.num_clusters]
            self.use_centroid_patients = True
            debug.trace(4, f"random prototypes: {self.cluster_patients}")

        # Read diagnostic data
        sc_diagnostics = SequenceCollector(csv_delimiter=self.input_delim)
        sc_diagnostics.read_patient_data(self.diagnostics_filename)
        top_diagnostics = []
        if self.top_diagnostics_filename:
            top_diagnostics = create_lookup_set(self.top_diagnostics_filename, self.max_codes)

        # Read prescription data
        sc_prescriptions = SequenceCollector(csv_delimiter=self.input_delim)
        sc_prescriptions.read_patient_data(self.prescriptions_filename)
        top_prescriptions = []
        if self.top_prescriptions_filename:
            top_prescriptions = create_lookup_set(self.top_prescriptions_filename, self.max_codes)
            
        # Read procedure data
        sc_procedures = SequenceCollector(csv_delimiter=self.input_delim)
        sc_procedures.read_patient_data(self.procedures_filename)
        top_procedures = []
        if self.top_procedures_filename:
            top_procedures = create_lookup_set(self.top_procedures_filename, self.max_codes)
            
        # Extract common IDs (e.g., patient IDs), convert series to binary [TODO: what], and perform cross correlation
        common_ids = list(patients_valid_index_date.keys())
        if REQUIRE_SEQUENCE_INTERSECTION:
            common_ids = system.intersection(sc_diagnostics.get_ids(),
                                             sc_prescriptions.get_ids(),
                                             sc_procedures.get_ids())
        tfe = TemporalFeatureExtractor(patients_valid_index_date, use_enumerated_codes=self.use_enumerated_codes, time_increment=self.time_increment)

        # Derive the temporal feature labels
        temporal_feature_labels = []
##      for  src in [DIAGNOSTICS_PREFIX, PRESCRIPTIONS_PREFIX]:
##          # The enumerated features have one value per type (e.g., s1-vs-s2 cross correlation)
##          temp_cross_correlation_labels = [(src + "-" + t) for t in ["s1-vs-s2", "s1-vs-s2", "s1-vs-s2"]]
##
##          # The binary features have a matrix per type
##          if not self.use_enumerated_codes:
##              for row in range(self.max_codes):
##                  for col in range(self.max_codes):
##                      for label in temp_cross_correlation_labels:
##                          temporal_feature_labels.append(label + "-" + temp_cross_correlation_labels[row, col] + "[_{r}-{c}]".format(r=row, c=col))
##          else:
##              temporal_feature_labels += temp_cross_correlation_labels
        ## TODO: add sequence-type prefix
        ## src1 = DIAGNOSTICS_PREFIX
        ## src2 = PRESCRIPTIONS_PREFIX
        # The enumerated features have one value per type (e.g., s1-vs-s2 cross correlation)
        ## OLD: temp_cross_correlation_labels = ["D1vsD1", "D1vsR2", "D1vsP3", "R2vsR2", "R2vsP3", "P3vsP3"]
        if INCLUDE_SELF_CORRELATIONS:
            CROSS_CORRELATION_LABELS = ["D1vD1", "D1vR2", "D1vP3", "R2vR2", "R2vP3", "P3vP3"]
    
            TYPE_LETTERS = ["D", "R", "P"]
            temp_cross_correlation_labels = []
            for i, letter1 in enumerate(TYPE_LETTERS):
                for j, letter2 in enumerate(TYPE_LETTERS):
                    if i <= j:
                        feature_label = "{l1}{i}v{l2}{j}".format(l1=letter1, i=(i + 1), l2=letter2, j=(j + 1))
                        temp_cross_correlation_labels.append(feature_label)
                        if self.use_enumerated_codes:
                            temporal_feature_labels.append(feature_label)
                        else:
                            # The binary features have a matrix per type
                            # TODO: generalize to more than 2 dimension (or drop)
                            for row in range(self.max_codes):
                                for col in range(self.max_codes):
                                    temporal_feature_labels.append("{lbl}{r}{c}".format(lbl=feature_label, r=row, c=col))
            debug.assertion(CROSS_CORRELATION_LABELS == temp_cross_correlation_labels)

        # Extract the sequences for the protoypes
        # TODO: have option to use random prototypes (to serve as a baseline)
        if self.use_centroid_patients:
            # Note: features labels are as follows: [DvD1c, RvR1c, PvP1c, ..., DvDKc, RvRKc, PvPKc]
            # where D is for diagnostics, R is for prescription, P is for procedures, c is for cluster,
            # and 1..K are the cluster numbers (1-based).
            for i in range(self.num_clusters):
                temporal_feature_labels += ("DvD{n}c RvR{n}c PvP{n}c".format(n=i + 1)).split()

            # Extract the temporal sequences for the cluster patients
            cluster_diagostics_sequence = [None] * (self.num_clusters)
            cluster_prescriptions_sequence = [None] * (self.num_clusters)
            cluster_procedures_sequence = [None] * (self.num_clusters)
            for i, cluster_patient_id in enumerate(self.cluster_patients):
                sequence = sc_diagnostics.get_data(cluster_patient_id)
                cluster_diagostics_sequence[i] = tfe.extract_temporal_sequences(cluster_patient_id, DIAGNOSTICS_PREFIX, sequence)
                sequence = sc_prescriptions.get_data(cluster_patient_id)
                cluster_prescriptions_sequence[i] = tfe.extract_temporal_sequences(cluster_patient_id, PRESCRIPTIONS_PREFIX, sequence)
                sequence = sc_procedures.get_data(cluster_patient_id)
                cluster_procedures_sequence[i] = tfe.extract_temporal_sequences(cluster_patient_id, PROCEDURES_PREFIX, sequence)


        # Output the feature table with values calculated on the fly (e.g., cross correlation against
        # the various cached temporal sequences for prototypes).
        csv_writer = csv.writer(sys.stdout, delimiter=self.output_delim)
        csv_writer.writerow([KEY_FIELD] + temporal_feature_labels)
        start_pos = self.starting_case - 1
        ## OLD
        ## num_cases = (self.num_cases if (self.num_cases >= 0) else (len(common_ids) - start_pos))
        ## debug.trace_fmt(5, "Using case range [s, e]", s=start_pos, e=(start_pos + num_cases - 1))
        ## for patient_id in common_ids[start_pos:num_cases]:
        num_cases = (self.num_cases if (self.num_cases >= 0) else max(0, (len(common_ids) - start_pos)))
        end_pos = (start_pos + num_cases)
        debug.trace_fmt(5, "Using case range [{s}, {e}]",
                        s=start_pos, e=(end_pos - 1))
        num_skipped = 0
        for patient_id in common_ids[start_pos:end_pos]:
            if (JUST_CENTROIDS and (patient_id not in self.cluster_patients)):
                debug.trace(4, f"Skipping non-centroid {patient_id}")
                num_skipped += 1
                continue
            debug.assertion(patient_id != KEY_FIELD)

            # Extract the sequences, resolving date strings into seconds
            # TODO: use cached value if prototype
            sequence = sc_diagnostics.get_data(patient_id)
            diagostics_sequence = tfe.extract_temporal_sequences(patient_id, DIAGNOSTICS_PREFIX, sequence)
            sequence = sc_prescriptions.get_data(patient_id)
            prescriptions_sequence = tfe.extract_temporal_sequences(patient_id, PRESCRIPTIONS_PREFIX, sequence)
            sequence = sc_procedures.get_data(patient_id)
            procedures_sequence = tfe.extract_temporal_sequences(patient_id, PROCEDURES_PREFIX, sequence)

            # Add features for cross correlations against self (6)
            # TODO: drop (n.b., used prior to addition of cluster-based features)
            temporal_feature_values = []
            if INCLUDE_SELF_CORRELATIONS:
                correlations = tfe.extract_cross_correlations(patient_id, diagostics_sequence, top_diagnostics, prescriptions_sequence, top_prescriptions, procedures_sequence, top_procedures)
                if self.use_enumerated_codes:
                    temporal_feature_values += [rnd(v) for v in correlations]
                else:
                    temporal_feature_values += [rnd(correlations[r, c]) for r in (self.max_codes + 1) for c in (self.max_codes + 1)]

            # Add features for cross correlations against (prototypical) cluster patients (3 x K)
            for i, cluster_patient_id in enumerate(self.cluster_patients):
                debug.trace(5, f"{patient_id} vs. prototype {cluster_patient_id}")
                correlations = tfe.extract_paired_cross_correlations(patient_id, diagostics_sequence, cluster_patient_id, cluster_diagostics_sequence[i], sequence_label="diagnostic")
                correlations += tfe.extract_paired_cross_correlations(patient_id, prescriptions_sequence, cluster_patient_id, cluster_prescriptions_sequence[i], sequence_label="prescription")
                correlations += tfe.extract_paired_cross_correlations(patient_id, procedures_sequence, cluster_patient_id, cluster_procedures_sequence[i], sequence_label="procedures")
                temporal_feature_values += [rnd(v) for v in correlations]

            # do sanity check on feature labels and values
            debug.trace_fmt(6, "temporal_feature_values: {tfv}", tfv=temporal_feature_values)
            debug.assertion(len(temporal_feature_values) == len(temporal_feature_labels))

            # Print features (and trace out values and types)
            all_feature_values = ([str(patient_id)] + temporal_feature_values)
            debug.trace_fmt(7, "feature types: {ft}", ft=[type(v) for v in all_feature_values])
            debug.trace_array(6, all_feature_values, "feature values")
            csv_writer.writerow(all_feature_values)

        # Sanity checks
        if JUST_CENTROIDS:
            debug.assertion(num_skipped < num_cases)
            debug.trace(4, f"{num_skipped} of {num_cases} cases skipped")

        return

if __name__ == '__main__':
    # TODO: copy trace_current_context into debug
    tpo.trace_current_context(level=tpo.QUITE_DETAILED)
    OLD_FIXED_ARGUMENTS = [VALID_INDEX_FILENAME, DIAGNOSTICS_FILENAME, PRESCRIPTIONS_FILENAME, PROCEDURES_FILENAME]
    OLD_TEXT_OPTIONS =  [
        (TOP_DIAGNOSTICS, "List of most common diagnostic codes"),
        (TOP_PRESCRIPTIONS, "List of most common prescription codes"),
        (TOP_PROCEDURES, "List of most common procedure codes"),
        ]
    fixed_arguments = []
    other_text_options = []
    if USE_OLD_STYLE_INPUT:
        fixed_arguments = OLD_FIXED_ARGUMENTS
        other_text_options = OLD_TEXT_OPTIONS
    else:
        fixed_arguments = [(SEQUENCE_DATA_FILENAMES, "String list with filenames for the sequence data files: shell glob patterns can be used")]
        other_text_options = [(VALID_INDEX_FILENAME, "Name of file giving last-valid date for each case"),
                              (AUX_DATA_FILENAME, "Name of file giving auxiliary data for output features (e.g., patient and/or claim info)"),
                              (CLASS_FILENAME, "Name of file giving class for each case")]
    app = Script(
        description=__doc__,
        # TODO: Add note to usage with SQL commands for extracting diagnostic, prescription and procedure data.
        skip_input=True,
        manual_input=True,
        boolean_options=[(ENUMERATED_CODES, "Use integral value for each code, rather than making binary")],
        int_options=[(MAX_CODES, "Maximum number of common codes used in cross correlation matrix (not apppicable for enumerated sequences))"),
                     (STARTING_CASE, "Number for case to start (i.e., line number excluding header)"),
                     (NUM_CASES, "Number of cases to process (e.g., number of patients)"),
                     (TIME_INCREMENT, "Number of seconds for unit of temporal expansion")],
        text_options=[INPUT_DELIM, OUTPUT_DELIM, INPUT_SORTED,
                      (CLUSTER_PATIENTS, "IDs of patients near center of their clusters")] + other_text_options,
        positional_options=fixed_arguments)
    app.run()
