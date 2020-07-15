#! /usr/bin/env python
#
# Script for running machine learning experiment given following input files:
#     base.csv                    training data (e.g., with column 1 or N for key)
#
# Notes:
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
# - Handle UTF-8 field names and those with double quotes.
#-------------------------------------------------------------------------------
# Portions Copyright (c) 2018-2019 John Snow Labs and UCB Pharma.
# Portions Copyright (c) 2020 ScrappyCito, LLC.
#

"""Runs experiment through various SciKit learn classifiers and TensorFlow dense neural networks (DNN). This includes special support for learning parameters, in particular the number of hidden units."""

# Standard library imports
import json
import os
import re
import sys
import warnings
import pickle

# Note: python 3.6+ format strings are used
assert((sys.version_info.major >= 3) and (sys.version_info.minor >= 6))

# External package exports
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, brier_score_loss
import tensorflow as tf
import tensorflow.contrib.learn as tfcl           # pylint: disable=no-name-in-module

# Local imports
import debug
import system
import glue_helpers as gh
import tpo_common as tpo

# Set general defaults
pd.options.display.max_columns = 500
warnings.simplefilter('ignore')
OUTPUT_CSV = system.getenv_bool("OUTPUT_CSV")
DEFAULT_SEED = -1
SEED = system.getenv_int("SEED", DEFAULT_SEED)

#...............................................................................

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
    debug.trace_fmt(7, "safe_int({n}, [{df}, {b}]) => {r})",
                    n=numeric, df=default_value, b=base, r=result)
    return result

def extract_string_list(text):
    """Extract list of values in TEXT, separated by whitespace and/or commas"""
    # EX: extract_string_list("1  2,3") => [1, 2, 3]
    # TODO: allow for values with embedded spaces
    normalized_text = re.sub(",", " ", text)
    result = re.split("  *", normalized_text)
    debug.trace(6, f"extract_string_list({text}) => {result}")
    return result

def extract_int_list(text):
    """Extract list of integers from tokens in TEXT"""
    # TODO: extract_int_list(text, dtype=int): """Extract list of number of DTYPE from tokens in TEXT"""
    result = [safe_int(token) for token in extract_string_list(text)]
    debug.trace(7, f"extract_int_list({text}) => {result}")
    return result

#...............................................................................
# Neural network for deep learning with tensorflow and Scikit-Learn wrapper.
# Based originally on
#    www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html.
#

DEFAULT_MAX_STEPS = 100
MAX_STEPS = system.getenv_int("MAX_STEPS", DEFAULT_MAX_STEPS)
DEFAULT_NUM_STEPS = None
NUM_STEPS = system.getenv_int("NUM_STEPS", DEFAULT_NUM_STEPS)

class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Class for deep learning neural network using tensorflow via sklearn compatible wrapper.
    Note: wrapper classes are in tensorflow/contrib/learn (formerly called skflow)."""
    # Notes:
    # - Not derived from tfcl.DNNClassifier due to need to invoke both the wrapper functions via SKCompat(self) and the underlying tensorflow methods. For issues, see
    #    https://github.com/tensorflow/tensorflow/issues/7287 (SKFLOW/TFLearn SKCompat does not properly implement SKLearn predict and predict_proba)
    # - Parameters: num_epochs, num_batches, steps, max_steps, number of hidden layers, number of nodes per hidden layer, activation functions, feature types, and TODO.
    # TODO: derive class name via introspection
    class_name = "NeuralNetworkClassifier"
    # TODO: add num_epochs, num_batches, etc.
    ## TEST: param_names = ["hidden_units", "feature_columns", "steps", "max_steps",
    ##                      "hidden_units1", "hidden_units2", "hidden_units3", "hidden_units4", "hidden_units5"]

    ## OLD: def __init__(self, hidden_units=None, feature_columns=None, steps=None, max_steps=None, **kwargs):
    def __init__(self, hidden_units=None, feature_columns=None, steps=None, max_steps=None,
                 hidden_units1=None, hidden_units2=None, hidden_units3=None, hidden_units4=None, hidden_units5=None, **kwargs):
        """Class constructor: creates instance with HIDDEN_UNITS and FEATURE_COLUMNS, along with sklearn wrapper class instance
        Note: To support optimization testing, the hidden layers can be specified via separate variables per hidden layer (e.g., hidden_units1=N1, hidden_units2=N2, ...)"""
        debug.trace_fmt(5, "{cl}.__init__(hidden_units={hu}, feature_columns={fc}, steps={s}, max_steps={ms}, [hu1={hu1}, hu2={hu1}, hu3={hu1}, hu4={hu1}, hu5={hu1}]) kwargs={kw}",
                        cl=self.class_name, hu=hidden_units, fc=feature_columns, s=steps, ms=max_steps,
                        hu1=hidden_units1, hu2=hidden_units2, hu3=hidden_units3, hu4=hidden_units4, hu5=hidden_units5, 
                        kw=kwargs)
        ## TEST: self.hidden_units1 = self.hidden_units2 = self.hidden_units3 = self.hidden_units4 = self.hidden_units5 = None
        if hidden_units1:
            # Note: overrides hidden_units from hidden_unitN's
            hidden_units = [safe_int(hidden_units1), safe_int(hidden_units2), safe_int(hidden_units3), safe_int(hidden_units4), safe_int(hidden_units5)]
        if (hidden_units is None):
            hidden_units = []
            n = 1
            while (True):
                # Check for next hidden_unitsN variable
                hidden_layer_var = f"hidden_units{n}"
                ## OLD: hidden_layer_text = re.sub(",", " ", kwargs.get(hidden_layer_var, ""))
                hidden_layer_nodes = kwargs.get(hidden_layer_var, "")
                if not hidden_layer_nodes:
                    break

                # Get number of nodes for current hidden layer
                ## OLD: hidden_layer = [safe_int(num) for num in re.split(" *", hidden_layer_text)]
                num_nodes = safe_int(hidden_layer_nodes)
                debug.trace(4, f"hidden layer {n}: {num_nodes}")
                hidden_units.append(num_nodes)
                setattr(self, hidden_layer_var, num_nodes)

                # Add var to list of parameter names for the instance
                ## TEST: debug.assertion(hidden_layer_var not in self.param_names)
                ## TEST: self.param_names.append(hidden_layer_var)
                n += 1
            debug.assertion(hidden_units)
        else:
            debug.assertion("hidden_units1" not in kwargs)
        if steps is None:
            steps = NUM_STEPS
        self.steps = steps
        if max_steps is None:
            max_steps = MAX_STEPS
        self.max_steps = max_steps
        self.hidden_units = hidden_units
        self.feature_columns = feature_columns
        self.hidden_units1 = hidden_units1
        self.hidden_units2 = hidden_units2
        self.hidden_units3 = hidden_units3
        self.hidden_units4 = hidden_units4
        self.hidden_units5 = hidden_units5
        
        ## OLD: self.tf_clf = tfcl.DNNClassifier(hidden_units, feature_columns, **kwargs)
        debug.assertion(self.feature_columns)
        self.tf_clf = tfcl.DNNClassifier(self.hidden_units, self.feature_columns)
        self.skl_clf = tf.contrib.learn.SKCompat(self.tf_clf)
        ## TODO: super(NeuralNetworkClassifier, self).__init__(hidden_units, feature_columns, **kwargs)
        return

    def get_params(self, deep=True):
        """Return list of parameters supported"""
        ## OLD: result = {param: getattr(self, param, None) for param in self.param_names}
        result = super(NeuralNetworkClassifier, self).get_params(deep=deep)
        debug.trace_fmt(5, "{cl}.get_params(deep={d}) => {r}",
                        cl=self.class_name, d=deep, r=result)
        return result
    
    def fit(self, training_x=None, training_y=None, max_steps=None, **kwargs):
        """Train classifier over TRAINING_X and TRAINING_Y, using MAX_STEPS (default to 100)"""
        debug.trace_fmt(5, "{cl}.fit(_, _); len(x)={xl} kw={kw}",
                        cl=self.class_name, xl=len(training_x), kw=kwargs)
        debug.assertion(training_x is not None)
        debug.assertion(training_y is not None)
        if max_steps is None:
            max_steps = self.max_steps
        fit_result = self.skl_clf.fit(training_x, training_y, max_steps=max_steps, **kwargs)
        debug.assertion(fit_result == self.skl_clf)
        result = self
        debug.trace_fmt(5, "{cl}.fit(x=_, y=_, ms={ms}, [kw={kw}]) => {r}",
                        cl=self.class_name, ms=max_steps, kw=kwargs, r=result)
        debug.trace_fmt(7, "\tx={{\n{x}\n}}\n\ty={{\n{y}\n}}",
                        x=training_x, y=training_y)
        return result

    def predict(self, sample):
        """Return predicted class for each SAMPLE (returning vector as in sklearn)"""
        result = list(self.tf_clf.predict_classes(sample))
        debug.assertion(len(np.array(result).shape) == 1)
        debug.trace_fmt(6, "{cl}.predict({s}) => {r}", cl=self.class_name, s=sample, r=result)
        return result

    ## def score(self, X, y, sample_weight=None):
    ##     """Return average accuracy of prediction of X vs Y"""
    ##     # TODO: factor in SAMPLE_WEIGHT
    ##     debug.assertion(sample_weight is None)
    ##     predictions = self.predict(X)
    ##     num_good = sum([(predictions[i] == c) for (i, c) in enumerate(y)])
    ##     result = (num_good / len(y))
    ##     debug.trace(6, f"{cl}.score({X}, {y}) => {result}")
    ##     return result

    def score(self, X_values, y_values, sample_weight=None):
        """Return average accuracy of prediction of X_values vs Y_values"""
        result = super(NeuralNetworkClassifier, self).score(X_values, y_values, sample_weight=sample_weight)
        debug.trace(5, f"{self.class_name}.score(_, _, _) => {result}")
        return result
    
    def predict_proba(self, sample):
        """Return probabilities of outcome classes predicted for each SAMPLE (returning matrix as in sklearn)"""
        tf_result = self.tf_clf.predict_proba(sample)
        debug.trace_fmt(6, "tf_result={r}", r=tf_result)
        result = np.array([row for row in tf_result])
        ## TODO:
        debug.assertion(len(result.shape) == 2)
        debug.trace_fmt(6, "{cl}.predict_proba({s}) => {r}",
                        cl=self.class_name, s=sample, r=result)
        return result

#...............................................................................

USE_LR = system.getenv_bool("USE_LR")        # logistic regression
LR_PENALTY = system.getenv_text("LR_PENALTY", 'l2')
## TODO: LR_PENALTY = system.getenv_bool("LR_PENALTY", None)   # TODO: l1:...; l2:...
USE_NB = system.getenv_bool("USE_NB")        # naive bayes
USE_NN = system.getenv_bool("USE_NN")        # neural network (deep learning)
USE_XGB_DEFAULT = (not (USE_LR or USE_NB or USE_NN))
USE_XGB = system.getenv_bool("USE_XGB",      # extreme gradient boost
                             USE_XGB_DEFAULT)
USE_GPUS = system.getenv_bool("USE_GPUS")
DEFAULT_HIDDEN_UNITS = extract_int_list(system.getenv_text("HIDDEN_UNITS", "10, 50"))
#
def create_classifier(feature_labels=None, training_X=None):
    """Create new sklearn classifier (defaulting to XGBoost), using optional FEATURE_LABELS and TRAINING_X for preprocessing.
    Note: FEATURE_LABELS and TRAINING_X are required for TensorFlow-based neural networks."""
    if feature_labels is None:
        feature_labels = []
    # TODO: redefine as a class
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
        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(training_X)
        # TODO: derive hidden unit configuration based on size of feature set
        ## OLD: new_clf = NeuralNetworkClassifier(hidden_units, feature_columns)
        new_clf = NeuralNetworkClassifier(hidden_units=hidden_units, feature_columns=feature_columns)
    else:
        debug.assertion(USE_XGB)
        misc_xgb_params = {}
        if not USE_GPUS:
            misc_xgb_params['n_gpus'] = 0
        debug.trace_var(6, 'misc_xgb_params')
        new_clf = xgb.XGBClassifier(**misc_xgb_params)
    debug.trace_fmt(5, "create_classifier() => {clf}", clf=new_clf)
    return new_clf

#...............................................................................
# TODO: rework inside of a function (e.g., main) to avoid variable clashes

# Note: tpo_common just used for showing elapsed time at end (if DEBUG_LEVEL 3+)
tpo.reference_variables(tpo)

# Check command line and show usage if requested
#
DEFAULT_BASE_NAME = "features"
DEFAULT_FIELD_SEP = ","
DEFAULT_GETENV_BOOL = system.DEFAULT_GETENV_BOOL
debug.assertion(not DEFAULT_GETENV_BOOL)
#
if ((len(sys.argv) > 1) and (sys.argv[1] == "--help")):
    print("")
    print("Usage: {script} [--help]".format(script=sys.argv[0]))
    print("")
    print("Run classification experiment given training data and classification column")
    print("")
    print("Environment variables:")
    print("")
    print("DATA_DIR:\tDirectory for data files (default $USER/home)")
    print("BASE_NAME:\tBase filename for experiment data (default {dbn})".format(dbn=DEFAULT_BASE_NAME))
    print("FIELD_SEP:\tString with value for field separator (default {dfs})".format(dfs=DEFAULT_FIELD_SEP))
    print("SEED:\tValue to use for random number seed override or -1 for n/a (default {ds})".format(ds=DEFAULT_SEED))
    print("RUN_OPTIMIZATION:\tWhether to run (gridsearch) optimization")
    # TODO: Add miscellaneous options (e.g., USE_LR, USE_NB, USE_XGB, and LR_PENALTY)
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
user = system.getenv_text("USER", None) or system.getenv_text("USERNAME", None) or "root"
home_dir = system.getenv_text("HOME", None) or system.getenv_text("USERPROFILE", None) or ("F:\\" + user)
train_data_dir = system.getenv_text("DATA_DIR", gh.form_path(home_dir, "data"))
basename = system.getenv_text("BASE_NAME", DEFAULT_BASE_NAME)
full_basename = os.path.join(train_data_dir, basename)
train_data_path = full_basename + ".csv"
data_sep = system.getenv_text("FIELD_SEP", DEFAULT_FIELD_SEP)    # delimiter

# Read in the all training data.
# Note: This gets pruned (e.g., via list of top features or via join with aux data).
# Also, the dtype is specified to avoid stupid pandas merge problem (with stupider suggestion):
#    You are trying to merge on int64 and object columns. If you wish to proceed you should use pd.concat.
all_train_data = pd.read_csv(train_data_path, sep=data_sep, dtype=str, encoding="UTF-8")

# Get training data proper
# Note: float conversion needed as input is string to avoid stupid pandas merge problem (see above)
X = top_features_data.astype(float)
y = train_labels[OUTCOME_VARIABLE].map({'Success': 1, 'Failure': 0,     # TDS
                                        'DRE': 1, 'Non_DRE': 0})        # DRE
# Optionally output the reduced training data
# Note: Extraneous separator added at front by pandas for index number.
top_basename = full_basename + "-top-features"
if OUTPUT_CSV:
    top_features_data.to_csv(top_basename + "-X.csv", sep=data_sep)
    y.to_csv(top_basename + "-y.csv", sep=data_sep)
    top_features_plus_data.to_csv(top_basename + ".csv", sep=data_sep)
debug.trace_fmt(7, "X = {{\n{X}\n}}\ny={{\n{y}\n}}", X=X, y=y)

roc_aucs = []
rocs = []
aps = []
brier_scores = []
train_counts = []
test_counts = []
precision_recall_curves = []
NUM_SPLITS = system.getenv_int("NUM_SPLITS", 3)
## kf = StratifiedKFold(n_splits=3, shuffle=True)
kf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True)

# Run classifier over each split
for train_index, test_index in kf.split(X, y):
    debug.trace_fmt(7, "train_index={trn}\ntest_index={tst}\n", trn=train_index, tst=test_index)
    train_counts.append(len(train_index))
    test_counts.append(len(test_index))
    try:
        clf = create_classifier(top_features, X.loc[train_index])
        debug.trace_fmt(7, "X={{\n{X}\n}}\ny={{\n{y}\n}}\n",
                        X=X.loc[train_index], y=y.loc[train_index])
        clf_model = clf.fit(X.loc[train_index], y.loc[train_index])
        predictions = clf_model.predict(X.loc[test_index])
        prediction_probs = clf_model.predict_proba(X.loc[test_index])[:, 1]
    except:
        debug.raise_exception(5)
        system.print_stderr("Exception during classification: {exc}", exc=sys.exc_info())
        break
    actuals = y.loc[test_index]
    print('Confusion matrix')
    print(confusion_matrix(actuals, predictions))
    print('ROC_AUC')
    roc_auc = roc_auc_score(actuals, prediction_probs)
    print(roc_auc)
    print('AP')
    ap = average_precision_score(actuals, prediction_probs)
    print(ap)
    aps.append(ap)
    print('Brier score: ')
    brier = brier_score_loss(actuals, prediction_probs)
    print(brier)
    print('(train, test) counts: ({trn}, {tst})'.format(trn=train_counts[-1], tst=test_counts[-1]))
    brier_scores.append(brier)
    roc_aucs.append(roc_auc)
    rocs.append(roc_curve(actuals, prediction_probs)[0:2])
    precision_recall_curves.append(precision_recall_curve(actuals, prediction_probs))

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
RANDOM_OPTIMIZATION = system.getenv_bool("RANDOM_OPTIMIZATION")
RUN_OPTIMIZATION = system.getenv_bool("RUN_OPTIMIZATION", RANDOM_OPTIMIZATION)
SCORING_METRIC = system.getenv_text("SCORING_METRIC", None)
SAVE_MODEL = system.getenv_bool("SAVE_MODEL")
if RUN_OPTIMIZATION:
    model_save_path = top_basename + ".model"

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
        hidden_units_values = [10, 50, 100, 250]
        max_steps_values = [10, 100, 1000]
        if RANDOM_OPTIMIZATION:
            hidden_units_values = [5, 10, 25, 50, 75, 100, 250, 500]
            max_steps_values = [10, 25, 50, 100, 250, 500, 1000]
        grid_search_params = {'hidden_units1': hidden_units_values,
                              'hidden_units2': hidden_units_values,
                              'max_steps': max_steps_values,
                              }
    else:
        system.print_stderr("Error: parameter grid for {typ} classifiers not yet defined", typ=type(clf_model))
    #
    if RANDOM_OPTIMIZATION:
        clf = RandomizedSearchCV(clf_model, grid_search_params, verbose=1, n_iter=100)
    else:
        clf = GridSearchCV(clf_model, grid_search_params, scoring=SCORING_METRIC, verbose=1)
    clf.fit(X, y)
    print(clf.best_score_)
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
