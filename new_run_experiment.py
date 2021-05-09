#! /usr/bin/env python
#
# Script for running epilepsy experiment given following input files:
#     base.csv                    training data
#     base-keys.csv               answers (from above)
#     base-top-features.json      subset of features to use
#
# Notes:
# - Based on Ojeksandr's DRE experiment, which reproduces a follow-up experiments by
#   UCB Pharma to the one described in the following paper:
#      An, S., et al. (2018), "Predicting drug-resistant epilepsy--A machine learning approach based on
#      administrative claims data", Epilepsy & Behavior 89: 118-125.
# - Includes support for the classificatin of the TDS (treatment decision system) experiment. See
#      Devinksy, O., et al. (2016), "Changing the approach to treatment choice in epilepsy using big data",
#      Epilepsy & Behavior 56: 32-37.
# - Main environment options:
#      BASE_NAME: name of datafile without .csv extension (e.g., JSLTrain90_TDS_Cohort)
#      SHOW_PLOT: produces plots
#      RUN_OPTIMIZATION: perform parameter optimization step
#      FIELD_SEP: separator used between fields
# - Other environment options:
#      KEYS_FILE: file specifiying key for training data
#      TOP_FEATURES: filename for json format list
#      DATA_DIR: where csv file reside
#
# TODO:
# - * Simplify trace_fmt via new f-string interpolation!
# - Handle UTF-8 field names and those with double quotes.
# - Set max_steps to a fixed value during debugging.
# - Add activation function to TF optimization (e.g.. relu or softmax).
#-------------------------------------------------------------------------------
#

"""Experiment for determining Drug-resistent Epilepsy (DRE) based on drugs taken and side effects. This optionally can be used for Treatment Decision System (TDS) experiments."""

# Standard library imports
import json
import os
import re
import sys
## OLD: import warnings
import pickle

# Note: python 3.6+ format strings are used
assert((sys.version_info.major >= 3) and (sys.version_info.minor >= 6))

# External package exports
import pandas as pd
import numpy
import numpy as np                       # pylint: disable=reimported
## OLD: import xgboost as xgb
## OLD: from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, brier_score_loss
## OLD: import tensorflow as tf
## import old_tensorflow as tf
## import old_tensorflow as tensorflow
## import tensorflow.contrib.learn as tfcl           # pylint: disable=no-member, no-name-in-module

# Local imports
import debug
import glue_helpers as gh
import system
import text_utils
from text_utils import version_to_number
from tf_dnn import NeuralNetworkClassifier
import tpo_common as tpo

# General settings
SAVE_MODEL = system.getenv_bool("SAVE_MODEL", False)
RANDOM_OPTIMIZATION = system.getenv_bool("RANDOM_OPTIMIZATION", False)
RUN_OPTIMIZATION = system.getenv_bool("RUN_OPTIMIZATION", RANDOM_OPTIMIZATION)
SCORING_METRIC = system.getenv_text("SCORING_METRIC", None)
QUICK_SEARCH = system.getenv_bool("QUICK_SEARCH", False)
AUX_TRAIN_DATA = system.getenv_text("AUX_TRAIN_DATA", "")
DEFAULT_VERBOSITY = ((debug.get_level() + 1) // 2)
VERBOSITY_LEVEL = system.getenv_int("VERBOSITY_LEVEL", DEFAULT_VERBOSITY)

# Optional modules
USE_LR = system.getenv_bool("USE_LR")        # logistic regression
USE_NB = system.getenv_bool("USE_NB")        # naive bayes
USE_NN = system.getenv_bool("USE_NN")        # neural network (deep learning)
USE_XGB_DEFAULT = (not (USE_LR or USE_NB or USE_NN))
USE_XGB = system.getenv_bool("USE_XGB",      # extreme gradient boost
                             USE_XGB_DEFAULT)
if USE_XGB:
    import xgboost as xgb
if USE_NN:
    import tensorflow as tf
    # note: Unfortunately, tensorflow dropped contrib in version 2.0.
    V1_13_1 = version_to_number("1.13.1")
    V2_0 = version_to_number("2.0")
    debug.assertion(V1_13_1 <= version_to_number(tf.version.VERSION) < V2_0)
    ## import tensorflow.contrib.learn as tfcl  # pylint: disable=no-member, no-name-in-module

# Enable optional plotting support
SHOW_PLOT = system.getenv_bool("SHOW_PLOT")
if SHOW_PLOT:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Make sure the output of plotting commands is displayed inline within frontends like the Jupyter notebook,
    # directly below the code cell that produced it. See following:
    #    https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline
    # Note: IPython magic commands yield syntax errors in regular python [WTH?]
    ## ORIGINAL: %matplotlib inline
    from IPython import get_ipython
    ## OLD: get_ipython().run_line_magic('matplotlib', 'inline')
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except AttributeError:
        system.print_stderr("Problem setting inline plots: {exc}", exc=sys.exc_info())
        
    # TODO: see if following redundant. It is for setting aesthetic parameters all at once,
    # but nothing is specified.
    #   set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set()

# Set general defaults
## OLD: pd.options.display.max_columns = 500
## OLD: warnings.simplefilter('ignore')
OUTPUT_CSV = system.getenv_bool("OUTPUT_CSV")
DEFAULT_SEED = -1
SEED = system.getenv_int("SEED", DEFAULT_SEED)
PATIENT_ID = "patient_id"
## OUTCOME_VARIABLE = 'Outcome_variable'
OUTCOME_VARIABLE_DEFAULT = "Outcome_variable"
OUTCOME_VARIABLE = system.getenv_text("OUTCOME_VARIABLE", OUTCOME_VARIABLE_DEFAULT)
UCB_DATA = (OUTCOME_VARIABLE == OUTCOME_VARIABLE_DEFAULT)
FULL_ROC = system.getenv_boolean("FULL_ROC", False)

FULLER_SEARCH = (RANDOM_OPTIMIZATION and not QUICK_SEARCH)
HIDDEN_UNITS_DEFAULTS = ("0 5 10 25 50 75 100 250 500" if FULLER_SEARCH else "0 10 50 100 250")
## MAX_STEPS_DEFAULTS = ("10 25 50 100 250 500 1000" if FULLER_SEARCH else "10 100 1000")
MAX_STEPS_DEFAULTS = ("100 1000 10000" if FULLER_SEARCH else "100 1000")
NUM_ITERS = system.getenv_int("NUM_ITERS", 1000)
NUM_HIDDEN_UNIT_VARS = system.getenv_int("NUM_HIDDEN_UNIT_VARS", 5)

#...............................................................................

def plot_curves(roc_curve_results, inverse_y=False):
    """Plot ROC_CURVE_RESULTS data, optionally with INVERSE_Y.
    Each roc_curve result is tuple: (true_positives, false_positives, thresholds).
    """
    for roc_curve_data in roc_curve_results:
        plt.plot(roc_curve_data[0], roc_curve_data[1])
    if inverse_y:
        plt.gca().invert_yaxis()
    plt.show()
    return


def is_symbolic(token):
    """Indicates whether token is symbolic (e.g., non-numeric)"""
    # EX: is_symbolic("PI") => True
    # EX: is_symbolic("3.14159") => False
    result = True
    try:
        result = token.strip() and (not float(token))
    except (TypeError, ValueError):
        pass
    debug.trace_fmt(7, "is_symbolic({t}) => {r})", t=token, r=result)
    return result

DEFAULT_PRECISION = 3
#
def rounded_mean(values, precision=DEFAULT_PRECISION):
    """Return the rounded mean of the numbers"""
    return system.round_num(np.mean(values), precision)


def safe_int(numeric, default_value=0, base=10):
    """Returns NUMERIC interpreted as a int, using DEFAULT_VALUE (0) if not
    numeric and interpretting value as optional BASE (10)"""
    try:
        result = int(numeric, base)
    except (TypeError, ValueError):
        debug.trace_fmt(4, "Warning: Exception converting integer numeric ({num}): {exc}",
                        num=numeric, exc=str(sys.exc_info()))
        result = default_value
    debug.trace_fmt(7, "safe_int({n}, [def={df}, base={b}]) => {r})",
                    n=numeric, df=default_value, b=base, r=result)
    return result

## OLD
## def extract_string_list(text):
##     """Extract list of values in TEXT, separated by whitespace and/or commas"""
##     # EX: extract_string_list("1  2,3") => [1, 2, 3]
##     # TODO: allow for values with embedded spaces
##     normalized_text = re.sub(",", " ", text)
##     result = re.split("  *", normalized_text)
##     debug.trace(6, f"extract_string_list({text}) => {result}")
##     return result

## OLD
## def extract_int_list(text):
##     """Extract list of integers from tokens in TEXT"""
##     # TODO: extract_int_list(text, dtype=int): """Extract list of number of DTYPE from tokens in TEXT"""
##     result = [safe_int(token) for token in extract_string_list(text)]
##     debug.trace(7, f"extract_int_list({text}) => {result}")
##     return result

def extract_string_list(text):
    """Extract list of string values in TEXT string separated by spacing or a comma"""
    # EX: extract_string_list("1, 2,3 4") => ['1', '2', '3', '4']
    # TODO: put into system.py
    normalized_text = text.replace(",", " ")
    value_list = re.split(" +", normalized_text)
    debug.trace_fmtd(5, "extract_string_list({t}) => {vl}", t=text, vl=value_list)
    return value_list

def extract_int_list(text, default_value=0):
    """Extract list of integral values from comma and/or whitespace delimited TEXT using DEFAULT_VALUE for non-integers (even if floating point)"""
    return [safe_int(v, default_value) for v in extract_string_list(text)]

def create_feature_mapping(label_values):
    """Return hash mapping elements from LABEL_VALUES into integers"""
    # EX: create_feature_mapping(['c', 'b, 'b', 'a']) => {'c':0, 'b':1, 'a':2}
    debug.assertion(isinstance(label_values, list))
    id_hash = {}
    for item in label_values:
        if (item not in id_hash):
            id_hash[item] = len(id_hash)
    debug.trace(7, f"create_feature_mapping({label_values}) => {id_hash}")
    return id_hash

#...............................................................................

## OLD: learning method put above to allow for conditional imports
## USE_LR = system.getenv_bool("USE_LR")        # logistic regression
LR_PENALTY = system.getenv_text("LR_PENALTY", 'l2')
## TODO: LR_PENALTY = system.getenv_bool("LR_PENALTY", None)   # TODO: l1:...; l2:...
## USE_NB = system.getenv_bool("USE_NB")        # naive bayes
## USE_NN = system.getenv_bool("USE_NN")        # neural network (deep learning)
## USE_XGB_DEFAULT = (not (USE_LR or USE_NB or USE_NN))
## USE_XGB = system.getenv_bool("USE_XGB",      # extreme gradient boost
##                              USE_XGB_DEFAULT)
USE_GPUS = system.getenv_bool("USE_GPUS", False)
DEFAULT_HIDDEN_UNITS = extract_int_list(system.getenv_text("HIDDEN_UNITS", "10, 50"))
#
def create_classifier(feature_labels=None, training_X=None):
    """Create new sklearn classifier (defaulting to XGBoost), using optional FEATURE_LABELS and TRAINING_X for preprocessing.
    Note: FEATURE_LABELS and TRAINING_X are required for TensorFlow-based neural networks."""
    if feature_labels is None:
        feature_labels = []
    # TODO: refedine as a class
    new_clf = None
    if USE_LR:
        new_clf = LogisticRegression(penalty=LR_PENALTY)
    elif USE_NB:
        new_clf = MultinomialNB()
    elif USE_NN:
        debug.assertion(feature_labels)
        debug.assertion(training_X is not None)
        ## TODO: put the feature preprocessing in the NeuralNetworkClassifier class
        hidden_units = DEFAULT_HIDDEN_UNITS
        ## OLD: feature_columns = [tf.feature_column.numeric_column(key=f) for f in feature_labels]
        ## TODO: let NeuralNetworkClassifier handle the feature inference
        ## OLD2:
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(training_X)  # pylint: disable=no-member, no-name-in-module
        # TODO: derive hidden unit configuration based on size of feature set
        ## OLD: new_clf = NeuralNetworkClassifier(hidden_units, feature_columns)
        ## OLD2:
        new_clf = NeuralNetworkClassifier(hidden_units=hidden_units, feature_columns=feature_columns)
        ## TODO: new_clf = NeuralNetworkClassifier(hidden_units=hidden_units)
    else:
        debug.assertion(USE_XGB)
        misc_xgb_params = {}
        if not USE_GPUS:
            misc_xgb_params['n_gpus'] = 0
        debug.trace_fmt(6, "misc_xgb_params={mxp}", mxp=misc_xgb_params)
        new_clf = xgb.XGBClassifier(**misc_xgb_params)
    debug.trace_fmt(5, "create_classifier() => {clf}", clf=new_clf)
    return new_clf


#...............................................................................

def main():
    """Entry point for script"""

    # Note: tpo_common just used for showing elapsed time at end (if DEBUG_LEVEL 3+)
    tpo.reference_variables(tpo)
    
    # Check command line and show usage if requested
    #
    DEFAULT_BASE_NAME = "JSLTrain90_DRE"
    DEFAULT_KEYS_AFFIX = "-keys.csv"
    DEFAULT_TOP_FEATURES_AFFIX = "-top-features.json"
    DEFAULT_FIELD_SEP = "|"
    DEFAULT_GETENV_BOOL = system.DEFAULT_GETENV_BOOL
    debug.assertion(not DEFAULT_GETENV_BOOL)
    #
    if ((len(sys.argv) > 1) and (sys.argv[1] == "--help")):
        print("")
        print("Usage: {script} [--help]".format(script=sys.argv[0]))
        print("")
        print("Run classification experiment for epsilepsy: Drug-Resistant Epsilepsy (DRE) or Treatment Decision System (TDS).")
        print("")
        print("Environment variables:")
        print("")
        print("DATA_DIR:\tDirectory for data files (default $USER/home)")
        print("BASE_NAME:\tBase filename for experiment data (default {dbn})".format(dbn=DEFAULT_BASE_NAME))
        print("KEYS_FILE:\tFilename for outcome key values (default $BASE_NAME{kfa})".format(kfa=DEFAULT_KEYS_AFFIX))
        print("TOP_FEATURES:\tFilename for list of top features (default $BASE_NAME{dtfa})".format(dtfa=DEFAULT_TOP_FEATURES_AFFIX))
        print("FIELD_SEP:\tString with value for field separator (default {dfs})".format(dfs=DEFAULT_FIELD_SEP))
        print("OUTPUT_CSV:\tWhether to output CSV files for top features")
        print("SEED:\tValue to use for random number seed override or -1 for n/a (default {ds})".format(ds=DEFAULT_SEED))
        print("SHOW_PLOT\tWhether to show plot of ROC and precision)")
        print("RUN_OPTIMIZATION:\tWhether to run (gridsearch) optimization")
        # TODO: Add miscellaneous options (e.g., USE_LR, USE_NB, USE_XGB, and LR_PENALTY)
        print("AUX_TRAIN_DATA:\tComma-separated list of filenames with auxiliary training data")
        print("USE_ALL_AUX_FEATURES:\tWhether to use all features in auxilary data files")
        print("")
        print("Notes:")
        print("- Boolean options default to {dgb} and are enabled unless value is 0 or False".format(dgb=DEFAULT_GETENV_BOOL))
        print("- XGBoost is used less USE_LR or USE_NB specified (logistic regression or naive bayes)")
        print("")
        sys.exit()
    
    # Set random seed if specified
    if (SEED != -1):
        np.random.seed(SEED)
    
    # Get miscellaneous options
    user = system.getenv_text("USER", None) or system.getenv_text("USERNAME", None) or "Common"
    home_dir = system.getenv_text("HOME", None) or system.getenv_text("USERPROFILE", None) or ("F:\\" + user)
    train_data_dir = system.getenv_text("DATA_DIR", gh.form_path(home_dir, "data"))
    ## XFER: train_data_dir = "f:\\tomohara\\data"
    ## TODO: train_data_dir = os.path.join("c:\", "Users", "Thomas.Ohara", "Documents")
    basename = system.getenv_text("BASE_NAME", DEFAULT_BASE_NAME)
    full_basename = os.path.join(train_data_dir, basename)
    train_data_path = full_basename + ".csv"
    # note: the key files include the patient ID, index date, and Outcome,
    # which is the way the [???]
    train_labels_path = system.getenv_text("KEYS_FILE", full_basename + DEFAULT_KEYS_AFFIX)
    top_features_path = system.getenv_text("TOP_FEATURES", full_basename + DEFAULT_TOP_FEATURES_AFFIX)
    data_sep = system.getenv_text("FIELD_SEP", DEFAULT_FIELD_SEP)    # delimiter
    
    # Read in the all training data.
    # Note: This gets pruned (e.g., via list of top features or via join with aux data).
    # Also, the dtype is specified to avoid stupid pandas merge problem (with stupider suggestion):
    #    You are trying to merge on int64 and object columns. If you wish to proceed you should use pd.concat.
    # TODO: Use dtype=float if no merging is needed (e.g., no auxilary data).
    ## BAD: CSV_DTYPE = float if (not AUX_TRAIN_DATA) else str
    ## TEST:
    ## CSV_DTYPE = None if (not AUX_TRAIN_DATA) else str
    ## debug.trace(4, f"csv type: {CSV_DTYPE}")
    ## all_train_data = pd.read_csv(train_data_path, sep=data_sep, dtype=CSV_DTYPE, encoding="UTF-8")
    CSV_DTYPE = None if (not AUX_TRAIN_DATA) else str
    all_train_data = pd.read_csv(train_data_path, sep=data_sep, dtype=CSV_DTYPE, encoding="UTF-8")
    all_features = system.difference(list(all_train_data.columns), [OUTCOME_VARIABLE])

    # Optionally remove specified fields from data
    # Note: this is for compatibility with iris_pandas_sklearn.py
    IGNORE_FIELDS = text_utils.extract_string_list(system.getenv_text("IGNORE_FIELDS", ""))
    if IGNORE_FIELDS:
        debug.assertion(not system.difference(IGNORE_FIELDS, all_features))
        try:
            all_train_data = all_train_data.drop(IGNORE_FIELDS, axis=1)
            all_features = system.difference(list(all_train_data.columns), [OUTCOME_VARIABLE])
        except:
            system.print_stderr("Problem dropping ignored fields: {exc}", exc=sys.exc_info())

    # Read in any auxiliary training files and combine via left outer join on patient ID.
    # Note: This is used for time-series derived features.
    # TODO: add function for comma-splitting file list; add .csv to basenames with extensions
    aux_train_files = [f for f in re.split(", *", AUX_TRAIN_DATA.strip()) if f]
    aux_features = []
    for aux_train_file in aux_train_files:
        aux_data = pd.read_csv(aux_train_file, sep=data_sep, dtype=str)
        debug.assertion(list(aux_data)[0] == PATIENT_ID)
        aux_features.append(list(aux_data)[1:])
    
        # Combine, preserving order of main datafile keys
        all_train_data = pd.merge(all_train_data, aux_data, how='left', on=PATIENT_ID).fillna(0)
        if all_train_data.empty:
            system.print_stderr("Error: Empty join with aux training file ({f})", f=aux_train_file)
            sys.exit()
    USE_ALL_AUX_FEATURES = system.getenv_bool("USE_ALL_AUX_FEATURES")
    
    # Read in the subset of features to use for the experiment.
    # Check whether auxiliary data files not incorporated (due to not being in top features).
    is_json = top_features_path.endswith(".json")
    top_features = []
    if system.file_exists(top_features_path):
        with open(top_features_path, 'r') as f:
            top_features = (json.load(f) if is_json else f.read().split("\n"))
            for i, aux_train_file in enumerate(aux_train_files):
                if USE_ALL_AUX_FEATURES:
                    top_features += aux_features[i]
                else:
                    if (not system.intersection(aux_features[i], top_features)):
                        system.print_stderr("Warning: no features from aux file {f} in top list", f=aux_train_file)
            debug.assertion(PATIENT_ID not in top_features)
            all_features = top_features
    else:
        debug.trace_fmt(4, "No top features file: {tfp}", tfp=top_features_path)
    
    # Note: add category to features for subsetting to produce entire csv file.
    # Also produce a version with the patient ID for debugging purposes (e.g., to dianose problems merging data from temporal_features_from_claims.py).
    if OUTPUT_CSV:
        cat = OUTCOME_VARIABLE
        top_features_plus_labels = [PATIENT_ID] + top_features + [cat]
    
    # Prune the data to match the desired features
    top_features_data = all_train_data
    top_features_plus_data = None
    top_features_data_keyed = None
    if top_features:
        try:
            debug.assertion(all([is_symbolic(v) for v in list(all_train_data)]))
            # note: #mf = #tf - #af', where af' is the subset of af in tf
            missing_features = system.difference(top_features, list(all_train_data))
            if missing_features:
                debug.trace_fmtd(2, "Warning: subsetting features not in data: {mf}", mf=missing_features)
                # note: #nt = #tf - #mf = #tf - (#tf - #af') = #af'
                top_features = system.difference(top_features, missing_features)
            top_features_data_keyed = all_train_data[[PATIENT_ID] + top_features]
            top_features_data = all_train_data[top_features]
            if OUTPUT_CSV:
                top_features_plus_data = all_train_data[top_features_plus_labels]
        except:
            system.print_stderr("Problem pruning top features: {exc}", exc=sys.exc_info())
    
    # Read list of classification labels if given in separate file, otherwise use outcome-variable field.
    if gh.non_empty_file(train_labels_path):
        train_labels = pd.read_csv(train_labels_path, sep=data_sep, dtype=str)
        if top_features:
            train_labels = pd.merge(train_labels, top_features_data_keyed[[PATIENT_ID]], how='inner', on=PATIENT_ID)
    else:
        ## TODO: specify OUTCOME_VARIABLE via command-line argument
        train_labels = all_train_data[[OUTCOME_VARIABLE]]
    
    # Get training data proper
    # Note: float conversion needed as input is string above to avoid stupid pandas merge problem (see above)
    # TODO: ** If merging skipped, input data as float above for better clarity.
    ## TEST: X = all_train_data[all_features].values TEST: y =
    ## train_labels.values
    ## OLD: X = all_train_data[all_features]
    ## OLD: y = train_labels
    X = y = None
    if UCB_DATA:
        X = top_features_data.astype(float)
        y = train_labels[OUTCOME_VARIABLE].map({'Success': 1, 'Failure': 0,     # TDS
                                                'DRE': 1, 'Non_DRE': 0})        # DRE
    else:
        # Convert features values to float and classification value to integer.
        # TODO: straighten out all_train_data vs. top_features_data usage,
        # as well as the confusing str vs. float usage.
        ## BAD: X = all_train_data.astype(float)
        ## BAD: X = all_train_data.values
        X = all_train_data[all_features].astype(float)
        train_labels_proper = train_labels[OUTCOME_VARIABLE]
        y = train_labels_proper.map(create_feature_mapping(train_labels_proper))

    # Optionally output the reduced training data
    # Note: Extraneous separator added at front by pandas for index number.
    all_basename = full_basename
    if top_features:
        top_basename = full_basename + "-top-features"
        if OUTPUT_CSV:
            top_features_data.to_csv(top_basename + "-X.csv", sep=data_sep)
            y.to_csv(top_basename + "-y.csv", sep=data_sep)
            top_features_plus_data.to_csv(top_basename + ".csv", sep=data_sep)
        all_basename = top_basename
    else:
        if OUTPUT_CSV:
            X.to_csv(all_basename + "-X.csv", sep=data_sep)
            y.to_csv(all_basename + "-y.csv", sep=data_sep)
    debug.trace_fmt(7, "X = {{\n{X}\n}}\ny={{\n{y}\n}}", X=X, y=y)
    
    roc_aucs = []
    rocs = []
    aps = []
    brier_scores = []
    train_counts = []
    test_counts = []
    precision_recall_curves = []
    # TODO: make default 5 or 10
    NUM_SPLITS = system.getenv_int("NUM_SPLITS", 3)
    ## kf = StratifiedKFold(n_splits=3, shuffle=True)
    kf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True)
    
    # Run classifier over each split
    for train_index, test_index in kf.split(X, y):
        debug.trace_fmt(7, "train_index={trn}\ntest_index={tst}\n", trn=train_index, tst=test_index)
        train_counts.append(len(train_index))
        test_counts.append(len(test_index))
        try:
            # Instantiate classifier
            # Note: Includes sanity check for clone quirk with NeuralNetworkClassifier triggered
            # during optimization testing.
            debug.trace_fmt(7, "X={{\n{X}\n}}\ny={{\n{y}\n}}\n",
                            X=X.loc[train_index], y=y.loc[train_index])
            clf = create_classifier(all_features, X.loc[train_index])
            if debug.verbose_debugging():
                debug.assertion(not clone(clf) is clf)
            # Train classifier and then run over test data
            clf_model = clf.fit(X.loc[train_index], y.loc[train_index])
            predictions = clf_model.predict(X.loc[test_index])
            prediction_probas = clf_model.predict_proba(X.loc[test_index])
            debug.trace(7, f"prediction_probas: {prediction_probas}")
            ## OLD: prediction_probs = prediction_probas[:, 1]
            # Note: each item of the list is now a 1x2 numpy array, instead
            # of previously being probabilities.
            debug.assertion(isinstance(prediction_probas, list) and (len(prediction_probas) > 0) and isinstance(prediction_probas[1], numpy.ndarray) and (prediction_probas[1].shape == (2,)))
            ## BAD: prediction_probs = [item[1] for item in prediction_probas]
            prediction_probs = [item[1] for item in list(prediction_probas)]
        except:
            debug.raise_exception(6)
            system.print_stderr("Exception during classification: {exc}", exc=sys.exc_info())
            break
        actuals = y.loc[test_index]
        debug.trace(7, f"actuals: {actuals}\npredictions: {predictions} \nprediction_probs: {prediction_probs}")
        debug.trace(7, f"actuals: {actuals}")
        print('Confusion matrix')
        print(confusion_matrix(actuals, predictions))
        # TODO: only perform ROC analysis for binary classification tasks
        print('ROC_AUC')
        ## OLD: roc_auc = roc_auc_score(actuals, prediction_probs)
        # note: With ovo, averages AUC over all possible pairwise combinations.
        ## OLD2: roc_auc = roc_auc_score(actuals, prediction_probs, multi_class="ovo")
        roc_auc = -1
        try:
            roc_auc = roc_auc_score(actuals, prediction_probs, multi_class="ovo")
        except np.AxisError:
            system.print_stderr("Exception during roc_auc_score: {exc}", exc=sys.exc_info())
        print(roc_auc)
        print('AP')
        ## OLD: ap = average_precision_score(actuals, prediction_probs)
        ap = -1
        try:
            ap = average_precision_score(actuals, prediction_probs)
        except (ValueError, TypeError):
            system.print_stderr("Exception during average_precision_score: {exc}", exc=sys.exc_info())
        print(ap)
        aps.append(ap)
        print('Brier score: ')
        ## OLD: brier = brier_score_loss(actuals, prediction_probs)
        brier = -1
        try:
            brier = brier_score_loss(actuals, prediction_probs)
        except (ValueError, TypeError):
            system.print_stderr("Exception during brier_score_loss: {exc}", exc=sys.exc_info())
        print(brier)
        print('(train, test) counts: ({trn}, {tst})'.format(trn=train_counts[-1], tst=test_counts[-1]))
        brier_scores.append(brier)
        roc_aucs.append(roc_auc)
        ## OLD: rocs.append(roc_curve(actuals, prediction_probs)[0:2])
        print('ROC curve: ')
        roc = -1
        try:
            # note: returns (trp, fpr, thresholds)
            roc = roc_curve(actuals, prediction_probs)
            debug.assertion(len(roc) == 3)
            if (not FULL_ROC):
                roc = roc[0:2]
        except (ValueError, TypeError):
            system.print_stderr("Exception during roc_curve: {exc}", exc=sys.exc_info())
        print(roc)
        rocs.append(roc)
        ## OLD: precision_recall_curves.append(precision_recall_curve(actuals, prediction_probs))
        print('Prec/Rec curve: ')
        prc = -1
        try:
            # note: returns (precision, recall, thresholds)
            prc = precision_recall_curve(actuals, prediction_probs)
        except (ValueError, TypeError):
            system.print_stderr("Exception during precision_recall_curve: {exc}", exc=sys.exc_info())
        print(prc)
        precision_recall_curves.append(prc)

    # Print average of statistics (AUC, etc.)
    print("Average ROC AUC score: " + str(rounded_mean(roc_aucs)))
    print("Average AP score: " + str(rounded_mean(aps)))
    print("Average Brier score: " + str(rounded_mean(brier_scores)))
    print("Average (train, test) counts: ({trn}, {tst})".
          format(trn=rounded_mean(train_counts, precision=1), tst=rounded_mean(test_counts, precision=1)))
    
    # Show ROC and precision/recall curves
    if SHOW_PLOT:
        plot_curves(rocs)
        plot_curves(precision_recall_curves)
    
    # Run optmization over parameter space, either brute force or probabilistic
    # TODO: use AUC as scoring metric
    if RUN_OPTIMIZATION:
        ## OLD: model_save_path = top_basename + ".model"
        model_save_path = all_basename + ".model"
    
        # Optimize parameters
        print("Parameter optimization")
        clf_model = create_classifier(top_features, X)
        grid_search_params = None
        if USE_XGB:
            grid_search_params = {'max_depth': [2, 4, 6],
                                  'n_estimators': [50, 100, 200]}
        elif USE_NN:
            # TODO: Base the range of hidden units on the number of features,
            # or allow for environment variable overrides (e.g, HIDDEN_UNITS1)
            # See https://www.heatonresearch.com/2017/06/01/hidden-layers.html.
            ## OLD: use hard-coded value
            ## hidden_units_values = [10, 50, 100, 250]
            ## max_steps_values = [10, 100, 1000]
            ## if RANDOM_OPTIMIZATION and not QUICK_SEARCH:
            ##     hidden_units_values = [5, 10, 25, 50, 75, 100, 250, 500]
            ##     max_steps_values = [10, 25, 50, 100, 250, 500, 1000]
            hidden_units_values = extract_int_list(system.getenv_text("HIDDEN_UNITS_VALUES", HIDDEN_UNITS_DEFAULTS))
            max_steps_values = extract_int_list(system.getenv_text("MAX_STEPS_VALUES", MAX_STEPS_DEFAULTS))
            hidden_unit_params = {"hidden_units{n}".format(n=(i + 1)): hidden_units_values
                                  for i in range(NUM_HIDDEN_UNIT_VARS)}
            grid_search_params = {'max_steps': max_steps_values}
            grid_search_params.update(hidden_unit_params)
        else:
            system.print_stderr("Error: parameter grid for {typ} classifiers not yet defined", typ=type(clf_model))
        debug.trace(4, f"grid_search_params: {grid_search_params}")
        #
        try:
            if RANDOM_OPTIMIZATION:
                # TODO: make verbose and n_iter values run-time options
                clf = RandomizedSearchCV(clf_model, grid_search_params, n_iter=NUM_ITERS,
                                         error_score=0, verbose=VERBOSITY_LEVEL)
            else:
                clf = GridSearchCV(clf_model, grid_search_params, scoring=SCORING_METRIC,
                                   error_score=0, verbose=VERBOSITY_LEVEL)
            clf.fit(X, y)
        except:
            debug.raise_exception(6)
            system.print_stderr("Exception during optimization: {exc}", exc=sys.exc_info())
        debug.trace_object(4, clf.cv_results_, "cv_results_")
        print("Best score:")
        print(clf.best_score_)
        print("Best params:")
        print(clf.best_params_)
        
        # Save the sklearn API model in pickle format
        # Note: must open in binary format to pickle
        # TODO: fix problem w/ embedded thread.RLock objects (for DNNClassifier)
        if SAVE_MODEL:
            print("Pickling sklearn API models")
            pickle.dump(clf, open(model_save_path, "wb"))
            # Note; sanity check that saved classifiers works like the original
            clf2 = pickle.load(open(model_save_path, "rb"))
            print(np.allclose(clf.predict(X), clf2.predict(X)))

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
