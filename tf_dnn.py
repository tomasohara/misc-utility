#! /usr/bin/env python
#
# Classifier for dense neural network based on tensorflow.
#
# Notes:
# - This was originally based on the sklearn compatible wrapper. See
#    www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html.
# - Important parameters for tf.estimator.DNNClassifier:
#   hidden_units: List defining number of hidden layers and numbers of hiddens units per layer.
#   max_steps: Number of total steps for which to train model.
#
# TODO:
# - All classification varirable to be given first, instead of assuming last.
# - Convert remaining trace_fmt[d] calls to trace w/ f"..." strings.
# - Finish support for optional use of tensorflow's float32 [WTH?]!
#

"""Tensorflow dense neural network classifier"""

# Standard packages
## TODO: import re
import sys

# Note: python 3.6+ format strings are used
assert((sys.version_info.major >= 3) and (sys.version_info.minor >= 6))

# Installed packages
import numpy
import pandas
from sklearn import datasets
from sklearn.base import BaseEstimator, ClassifierMixin
## DEBUG:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.contrib.learn as tfcl  # pylint: disable=no-name-in-module, import-error

# Local packages
import debug
import system
from text_utils import extract_int_list, is_symbolic, version_to_number

V1_13_1 = version_to_number("1.13.1")
debug.assertion(V1_13_1 <= version_to_number(tf.version.VERSION))
DEFAULT_HIDDEN_UNITS = extract_int_list(system.getenv_text("HIDDEN_UNITS", "20, 30"))
DEFAULT_MAX_STEPS = 100
MAX_STEPS = system.getenv_int("MAX_STEPS", DEFAULT_MAX_STEPS)
## TODO: drop NUM_STEPS (as no longer used)
DEFAULT_NUM_STEPS = None
NUM_STEPS = system.getenv_int("NUM_STEPS", DEFAULT_NUM_STEPS)
## TODO:
## FULLER_SEARCH = (RANDOM_OPTIMIZATION and not QUICK_SEARCH)
## HIDDEN_UNITS_DEFAULTS = ("5 10 25 50 75 100 250 500" if FULLER_SEARCH else "10 50 100 250")
## MAX_STEPS_DEFAULTS = ("10 25 50 100 250 500 1000" if FULLER_SEARCH else "10 100 1000")
USE_FLOAT32 = system.getenv_bool("USE_FLOAT32", False)

#...............................................................................

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

MAX_HIDDEN_UNIT_VARS = system.getenv_int("MAX_HIDDEN_UNIT_VARS", 5)

class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """Class for deep learning neural network using tensorflow via sklearn compatible wrapper.
    In normal usage, the hidden units are specified via the hidden_units vector;
    however, to faciliate grid search, they can be specified by individual hidden_unitsN members.
    Note: wrapper classes are in tensorflow/contrib/learn (formerly called skflow)."""
    # Notes:
    # - Not derived from tfcl.DNNClassifier due to need to invoke both the wrapper functions via SKCompat(self) and the underlying tensorflow methods. For issues, see
    #    https://github.com/tensorflow/tensorflow/issues/7287 (SKFLOW/TFLearn SKCompat does not properly implement SKLearn predict and predict_proba)
    # - Parameters: num_epochs, num_batches, steps, max_steps, number of hidden layers, number of nodes per hidden layer, activation functions, feature types, and TODO.
    # TODO: derive class name via introspection; ** move feature description from fit to constructor to streamline optmization testing (and minimize tracing)
    class_name = "NeuralNetworkClassifier"
    # TODO: add num_epochs, num_batches, etc.
    ## TEST: param_names = ["hidden_units", "feature_columns", "steps", "max_steps",
    ##                      "hidden_units1", "hidden_units2", "hidden_units3", "hidden_units4", "hidden_units5"]
    ## OLD: MAX_HIDDEN_UNIT_VARS = 5
    HIDDEN_UNIT_VARS = ["hidden_units" + str(v + 1) for v in range(MAX_HIDDEN_UNIT_VARS)]

    def __init__(self, hidden_units=None, feature_columns=None, steps=None, max_steps=None, feature_names=None,
                 **kwargs):
        """Class constructor: creates instance with HIDDEN_UNITS and FEATURE_COLUMNS, along with sklearn wrapper class instance"""
        # TODO: do a sanity check for hidden units variables above the max (e.g., 10)
        debug.trace_fmt(5, "{cl}.__init__(hidden_units={hu}, feature_columns={fc}, steps={s}, max_steps={ms}) kwargs={kw}",
                        cl=self.class_name, hu=hidden_units, fc=feature_columns, s=steps, ms=max_steps,
                        kw=kwargs)
        # Supply default for hidden units, derived from hidden_unitsN's values if any non-zero
        ## OLD: if (hidden_units is None):
        # 
        alt_hidden_units = [(kwargs.get(v) or 0) for v in self.HIDDEN_UNIT_VARS]
        if (sum(alt_hidden_units) > 0):
            hidden_units = alt_hidden_units
        if hidden_units is None:
            hidden_units = DEFAULT_HIDDEN_UNITS
        if steps is None:
            steps = NUM_STEPS
        self.steps = steps
        if max_steps is None:
            max_steps = MAX_STEPS
        self.max_steps = max_steps
        ## OLD: self.hidden_units = hidden_units
        self.hidden_units = hidden_units[:]
        self.feature_columns = feature_columns
        self.feature_names = feature_names
        
        ## OLD: self.tf_clf = tfcl.DNNClassifier(self.hidden_units, self.feature_columns)
        ## OLD: self.skl_clf = tfcl.SKCompat(self.tf_clf)
        # Note: the underlying object aren't created until the data is passed in via fit method.
        self.tf_clf = self.skl_clf = None
        ## TODO: super(NeuralNetworkClassifier, self).__init__(hidden_units, feature_columns, **kwargs)
        return

    def get_params(self, deep=True):
        """Return list of parameters supported"""
        debug.trace_fmtd(7, "{cl}.get_params(deep={d})", cl=self.class_name, d=deep)
        result = super(NeuralNetworkClassifier, self).get_params(deep=deep)
        ## BAD: result += self.HIDDEN_UNIT_VARS
        result.update({v:getattr(self, v, None) for v in self.HIDDEN_UNIT_VARS})
        debug.trace_fmtd(5, "{cl}.get_params() => {r}",
                         cl=self.class_name, r=result)
        return result
    
    def fit(self, training_x=None, training_y=None, max_steps=None, **kwargs):
        """Train classifier over TRAINING_X and TRAINING_Y, using MAX_STEPS (default to 100)"""
        # TODO: isolate the feature preprocessing support in a separate methiod
        debug.trace_fmt(5, "{cl}.fit(_, _); len(x)={xl} kw={kw}",
                        cl=self.class_name, xl=len(training_x), kw=kwargs)
        if max_steps is None:
            max_steps = self.max_steps
        debug.assertion(training_x is not None)
        debug.assertion(training_y is not None)

        # Derive feature names from input.
        # Note: if no feature names are specified, pandas dataframes should be used.
        # This avoids having to use generic feature names
        is_pandas_dataframe = isinstance(training_x, pandas.core.frame.DataFrame)
        num_features = 0
        if self.feature_names is None:
            debug.assertion(is_pandas_dataframe)
            if is_pandas_dataframe:
                num_features = len(list(training_x.values[0, :]))
                self.feature_names = list(training_x.columns[0:-1])
            else:
                num_features = len(list(training_x[0, :]))
                self.feature_names = ["feature{c}".format(c=(i + 1)) for i in range(num_features)]
                debug.trace(2, f"Warning: generated generic feature names: {self.feature_names}")
            debug.assertion(isinstance(self.feature_names, list) and (len(self.feature_names) > 0))

        ## OLD: # Make sure training data is an array
        ## OLD: if (not isinstance(training_x, numpy.ndarray)):
        ## OLD:     debug.trace(5, "Converting training data to numpy array")
        ## OLD:     training_x = numpy.array(training_x)
        # Create training data feature descriptions (stupid brain-dead TensorFlow interface)
        is_numeric_field = [False] * num_features
        if self.feature_columns is None:
            ## OLD: # Convert strings to floating point if needed (effing unuseful TensorFlow!)
            ## # TODO: check all rows, not just the first
            ## if all([is_symbolic(v) for v in training_x[0, :]]):
            ##    debug.trace(5, "Converting strings in training data to floats")
            ##    training_x = training_x.astype(float)
            
            ## ## BAD: self.feature_columns = tfcl.infer_real_valued_columns_from_input(training_x)
            self.feature_columns = []
            for c in range(num_features):
                # TODO: WTH isn't this part of tensorflow?
                training_matrix = training_x.values if is_pandas_dataframe else training_x
                column_data = training_matrix[:, c]
                feature_column = tf.feature_column.numeric_column(self.feature_names[c])
                if any([is_symbolic(v) for v in column_data]):
                    feature_values = list(set(column_data))
                    feature_column = tf.feature_column.categorical_column_with_vocabulary_list(self.feature_names[c], feature_values)
                else:
                    is_numeric_field[c] = True
                self.feature_columns.append(feature_column)
            debug.trace(4, f"inferred feature_columns: {self.feature_columns}")
        debug.assertion(self.feature_columns)

        # Make sure the class values are in range 0..n-1 (stupid Tensorflow restriction!)
        class_values = list(training_y.values) if is_pandas_dataframe else list(training_y)
        class_hash = create_feature_mapping(class_values)
        num_classes = len(class_hash)
        encoded_classes = [class_hash[v] for v in list(class_values)]
        
        # Note: Optionally uses float32 given stupid TensorFlow quirk
        if USE_FLOAT32:
            training_x = training_x.copy()
            for c in range(num_features):
                if is_numeric_field[c]:
                    training_x[:, c] = training_x[:, c].asfloat(tf.float32)
            
        # TODO: do a deep copy to avoid stupid sklearn problem with parameters (e.g., hidden_units)
        try:
            self.tf_clf = tfcl.DNNClassifier(self.hidden_units, self.feature_columns,
                                             n_classes=num_classes)
            self.skl_clf = tfcl.SKCompat(self.tf_clf)
            fit_result = self.skl_clf.fit(training_x, encoded_classes, max_steps=max_steps, **kwargs)
        except:
            debug.trace_fmtd(2, "Error: Problem during fit: {exc}", exc=sys.exc_info())
            debug.raise_exception(6)

        debug.assertion(fit_result == self.skl_clf)
        result = self
        debug.trace_fmt(5, "{cl}.fit(x=_, y=_, ms={ms}, [kw={kw}]) => {r}",
                        cl=self.class_name, ms=max_steps, kw=kwargs, r=result)
        debug.trace_fmt(7, "\tx={{\n{x}\n}}\n\ty={{\n{y}\n}}",
                        x=training_x, y=training_y)
        return result

    def predict(self, sample):
        """Return predicted class for each SAMPLE (returning vector as in sklearn)"""
        debug.trace_fmt(7, "{cl}.predict({s})", cl=self.class_name, s=sample)
        result = list(self.tf_clf.predict_classes(sample))
        debug.assertion(len(numpy.array(result).shape) == 1)
        debug.trace_fmt(6, "{cl}.predict() => {r}", cl=self.class_name, s=sample, r=result)
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

    ## OLD: renamed args for the sake of pylint
    ## def score(self, X, y, sample_weight=None):
        ## """Return average accuracy of prediction of X_values vs Y_values"""
        ## OLD: result = super(NeuralNetworkClassifier, self).score(X_values, y_values, sample_weight=sample_weight)
        ## debug.trace(5, f"{self.class_name}.score(_, _, _) => {result}")
        ## return result

    def score(self, X, y, sample_weight=None):
        """Return average accuracy of prediction of X matrix vs Y vector"""
        debug.trace(6, f"{self.class_name}.score(_, _, [sw={sample_weight}]")
        result = super(NeuralNetworkClassifier, self).score(X, y, sample_weight=sample_weight)
        debug.trace(5, f"{self.class_name}.score() => {result}")
        return result

    def predict_proba(self, sample):
        """Return probabilities of outcome classes predicted for each SAMPLE (returning matrix as in sklearn)"""
        debug.trace(7, f"{self.class_name}.score()")
        tf_result = self.tf_clf.predict_proba(sample)
        debug.trace(6, f"predict_proba: {tf_result}")
        result = numpy.array(tf_result)
        ## TODO: debug.assertion(len(result.shape) == 2)
        ## OLD: # Convert result to list if very detailed tracing
        ## OLD: if debug.debugging(7):
        ##      ## BAD: result = list(result)
        ##      ## BAD2: result = result.tolist()
        ##      result = list(result.tolist())
        # Convert result to list
        result = list(result.tolist())
        debug.trace_fmt(6, "{cl}.predict_proba({s}) => {r}",
                        cl=self.class_name, s=sample, r=result)
        return result

#...............................................................................

def main():
    """Entry point for script
    Note:
    - This is just indented as a quick test to ensure basic functionality working.
      -- The data is based on randonly generated independent Gaussian features.
      -- Use NUM_SAMPLES and SEED env. vars to customize.
    - By default, this runs Tensorflow deep neural network over aritificial data.
      -- Set USE_LOGIT and USE_IRIS env. vars to use logistic regression and/or Iris data.
    """
    # Note: generate artificial classification task from (Hastie et al 2009, p339):
    # - ten independent Gaussian features
    # - y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
    # - where 9.34 is median of chi-square with 10 degrees of freedom
    # Ref: T. Hastie, R. Tibshirani and J. Friedman (2009), Elements of Statistical Learning (2nd Edition), Springer.
    #
    system.print_stderr("Warning: not intended for standalone usge; a simple test follows")

    # Derive the dataset
    NUM_SAMPLES = system.getenv_int("NUM_SAMPLES", 1000)
    SEED = system.getenv_int("SEED", 7919)
    ## DEBUG: digits = datasets.load_digits();  X = digits.data;  y = digits.target
    USE_IRIS = system.getenv_bool("USE_IRIS", False)
    USE_LOGIT = system.getenv_bool("USE_LOGIT", False)
    X = y = None
    if USE_IRIS:
        iris = datasets.load_iris()      
        X = iris.data                    # pylint: disable=no-member
        y = iris.target                  # pylint: disable=no-member
    else:
        (X, y) = datasets.make_hastie_10_2(n_samples=NUM_SAMPLES, random_state=SEED)
    debug.trace_fmtd(7, "X={{\n{X}\n}}\ny={{\n{y}\n}}", X=X, y=y)

    # Perform the classification and report the results
    try:
        clf = None
        if USE_LOGIT:
            clf = NeuralNetworkClassifier()
        else:
            clf = LogisticRegression()
        clf.fit(X, y)
        predictions = clf.predict(X)
        print("Accuracy: {a}".format(a=accuracy_score(y, predictions)))
        print("Confusion:\n{c}".format(c=confusion_matrix(y, predictions)))
        print("Report:\n{r}".format(r=classification_report(y, predictions)))
    except:
        debug.trace_fmtd(2, "Error: Problem during classification, etc.: {exc}",
                         exc=sys.exc_info())
        debug.raise_exception(6)
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
