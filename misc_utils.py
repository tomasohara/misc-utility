# Miscellaneous functions not suitable for other modules (e.g., system.py).
#
#
# TODO:
# - Separate list related functions (e.g., as list_utils.py).
#

"""Misc. utility functions"""

import inspect
import math
import re
import sys

import debug
import system
import text_utils

def transitive_closure(edge_list):
    """Computes transitive close for graph given by EDGE_LIST (i.e., makes indirect links explicit)"""
    # ex: transitive_closure([(1,2),(2,3),(3,4)]) => set([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (2, 4)])
    # notes; based on https://stackoverflow.com/questions/8673482/transitive-closure-python-tuples
    closure = set(edge_list)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)

        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break

        closure = closure_until_now

    return closure


def read_tabular_data(filename):
    """Reads table with (unique) key and tab-separated value. 
    Note: key made lowercase"""
    debug.trace_fmtd(4, "read_tabular_file({f})", f=filename)
    table = {}
    with open(filename) as f:
        for (i, line) in enumerate(f):
            line = system.from_utf8(line)
            items = line.split("\t")
            if len(items) == 2:
                assert(items[0].lower() not in table)
                table[items[0].lower()] = items[1]
            else:
                debug.trace_fmtd(4, "Ignoring item w/ unexpected format at line {num}",
                                 num=(i + 1))
    ## debug.trace_fmtd(7, "table={t}", t=table)
    debug.trace_values(7, table, "table")
    return table


def extract_string_list(text):
    """Extract a list of values from text using spaces and/or commas as delimiters"""
    # EX: extract_string_list("1  2,3") => [1, 2, 3]
    trimmed_text = re.sub("  +", " ", text.strip())
    values = trimmed_text.replace(" ", ",").split(",")
    debug.trace_fmtd(5, "extract_string_list({t}) => {v}", t=text, v=values)
    return values


def is_prime(num):
    """Moderately efficient function for testing whether a number is prime
    Notes:
    - Based on https://en.wikipedia.org/wiki/Primality_test.
    - The intuition is that primes > 3 are of form (6k +/- 1).
    """
    ## EX: FIRST_100_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]; (len(FIRST_100_PRIMES) == 100)
    ## EX: all([is_prime(n) for n in FIRST_100_PRIMES])
    ## EX: all([(not is_prime(n)) for n in range(FIRST_100_PRIMES[-1])  if n not in FIRST_100_PRIMES])
    ##
    debug.trace_fmt(5, "in is_prime({n})", n=num)
    is_prime_num = True

    # First, check primes below 4 (only 2)
    if (num <= 3):
        is_prime_num = (num > 1)
        if not is_prime_num:
            debug.trace_fmt(4, "{n} not prime as less than 3 and not 2.", n=num)

    # Next, make sure not divisible by 2 or 3
    elif ((num % 2 == 0) or (num % 3 == 0)):
        debug.trace_fmt(4, "{n} not prime as divisible by 2 or 3.", n=num)
        is_prime_num = False

    # Otherwise, check (6k +/- 1) values to see if divisible by 2 or 3,
    # stopping when value exceeds sqrt(n).
    last_possible = math.ceil(math.sqrt(num))
    i = 5
    while (i <= last_possible):
        if (((num % i) == 0) or ((num % (i + 2)) == 0)):
            debug.trace_fmt(4, "{n} not prime as (6k +/- 1) divisible by 2 or 3 for some k.", n=num)
            is_prime_num = False
            break
        i += 6

    debug.trace_fmt(5, "is_prime({n}) => {ip}", n=num, ip=is_prime_num)
    return is_prime_num


def fibonacci(max_num):
    """Returns Fibonacci sequence with numbers less than MAX_NUM"""
    # EX: fibonacci(10) => [0, 1, 1, 2, 3, 5, 8]
    a, b = 0, 1
    sequence = []
    while (a < max_num):
        sequence.append(a)
        a, b = b, (a + b)
    debug.trace_fmt(9, "fibonacci({m}) => {s}", m=max_num, s=sequence)
    return sequence


def sort_weighted_hash(weights, max_num=None, reverse=None):
    """sorts the entries in WEIGHTS hash, returns list of (key, freq) tuples.
    Note: sorted in REVERSE order by default"""
    if max_num is None:
        max_num = len(weights)
    if reverse is None:
        reverse = True
    sorted_keys = sorted(weights.keys(), reverse=reverse, 
                         key=lambda k: weights[k])
    top_values = [(k, weights[k]) for k in sorted_keys[:max_num]]
    debug.trace_fmt(5, "sort_weighted_hash(_, [max={m}], [rev={r}) => {t}",
                    m=max_num, r=reverse, t=top_values)
    return top_values


def unzip(iterable, num=None):
    """"Inverse of zip operation: returns n lists of i-th elements of input list of tuples. The optional NUM argument ensures that that many values returned (in case of empty input lists)"""
    # See https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip.
    # EX: unzip(zip([1, 2, 3], ['a', 'b', 'c'])) => [[1, 2, 3], ['a', 'b', 'c']]
    # EX: unzip(zip([], []), 2) => [[], []]
    result = [list(tupl) for tupl in zip(*iterable)]
    if not result and num:
        result = [[] for _i in range(num)]
    return result


def get_current_frame():
    """Return stack frame"""
    frame = inspect.currentframe().f_back
    debug.trace_fmt(6, "get_current_frame() => {f}", f=frame)
    return frame


def eval_expression(expr_text, frame=None):
    """Evaluate EXPRESSION_given_by_TEXT"""
    # EX: eval_expression("len([123, 321]) == 2")
    result = None
    try:
        if frame is None:
            frame = get_current_frame()
            # Note: need to get caller's frame
            frame = frame.f_back
        # pylint: disable=eval-used
        result = eval(expr_text, frame.f_globals, frame.f_locals)
    except:
        debug.trace_fmt(5, "Exception during eval_expression({expr}): {exc}",
                        exp=expr_text, exc=sys.exc_info())
    debug.trace_fmt(7, "eval_expression({expr}) => {r}",
                    expr=expr_text, r=result)
    return result


def trace_named_object(level, object_name, caller_frame=None):
    """Trace OBJECT_given_by_NAME"""
    # EX: trace_named_object(4, "sys.argv")
    if caller_frame is None:
        caller_frame = get_current_frame().f_back
    debug.trace_object(level, eval_expression(object_name,
                                              frame=caller_frame),
                       label=object_name)
    return


def trace_named_objects(level, list_text):
    """Trace objects in LIST_text"""
    # EX: trace_named_object(4, "[len(sys.argv), sys.argv]")
    debug.assertion(re.search("^\[.*\]$", list_text))
    frame = get_current_frame().f_back
    for name in text_utils.extract_string_list(list_text[1:-1]):
        trace_named_object(level, name, caller_frame=frame)
    return


#-------------------------------------------------------------------------------

def main(args):
    """Supporting code for command-line processing"""
    debug.trace_fmtd(6, "main({a})", a=args)
    system.print_stderr("Warning: not intended for direct invocation")
    return

if __name__ == '__main__':
    main(sys.argv)
