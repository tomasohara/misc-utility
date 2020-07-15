#! /usr/bin/env python
#
# Class for supervised classification using Scikit-Learn. See tutorial at
#    TODO: http://scikit-learn.org/stable/tutorial/<supervised-classification>
#
# NOTE:
# - This is a quick-n-dirty prototype created by converting the text
#   classifer supporting various SciKit-Learn method:
#      https://github.com/tomasohara/text-categorization
#
# TODO:
# - Upgrade to more recent version of Scikit Learn (e.g., 0.20).
# - Maintain cache of classification results.
# - Review classification code and add examples for clarification of parameters.
# - Fix SHOW_REPORT option for training.
#

"""Supervised classification support"""

# Standard packages
import sys

# Installed packages
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Local packages
import debug
import system
from system import print_stderr

# Common options
#
OUTPUT_BAD = system.getenv_bool("OUTPUT_BAD", False)
VERBOSE = system.getenv_bool("VERBOSE", False)
CLASS_FIRST = system.getenv_bool("CLASS_FIRST", False)
FIELD_SEP = system.getenv_text("FIELD_SEP", "\t")
EPSILON = system.getenv_float("EPSILON", 1e-6)

# Options for Support Vector Machines (SVM)
#
# Descriptions of the parameters can be found at following page:
#    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# note: defaults used for parameters (n.b., the value None is not usable due to
# sklearn constructor limitations).
USE_SVM = system.getenv_bool("USE_SVM", False)
SVM_KERNEL = system.getenv_text("SVM_KERNEL", "rbf")
SVM_PENALTY = system.getenv_float("SVM_PENALTY", 1.0)
SVM_MAX_ITER = system.getenv_int("SVM_MAX_ITER", -1)
SVM_VERBOSE = system.getenv_bool("SVM_VERBOSE", False)

# Options for Stochastic Gradient Descent (SGD)
#
# Descriptions of the parameters can be found at following page:
#    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# TODO: initialize to None and override only if non-Null
USE_SGD = system.getenv_bool("USE_SGD", False)
SGD_LOSS = system.getenv_text("SGD_HINGE", "hinge")
SGD_PENALTY = system.getenv_text("SGD_PENALTY", "l2")
SGD_ALPHA = system.getenv_float("SGD_ALPHA", 0.0001)
SGD_SEED = system.getenv_float("SGD_SEED", None)
SGD_MAX_ITER = system.getenv_int("SGD_MAX_ITER", 5)
SGD_TOLERANCE = system.getenv_float("SGD_TOLERANCE", None)
SGD_VERBOSE = system.getenv_bool("SGD_VERBOSE", False)

# Options for Logistic Regression (LR)
# See https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression:
#    class sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
#
# TODO: allow None for default (see note on constructor limitations above)
#
USE_LR = system.getenv_bool("USE_LR", False)
LR_PENALTY = system.getenv_text("LR_PENALTY", "l2")

# Options for Random Forest (RD)
# See https://scikit-learn.org/0.15/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier:
#    class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)
#
USE_RF = system.getenv_bool("USE_RF", False)
RF_NUM_ESTIMATORS = system.getenv_integer("RF_NUM_ESTIMATORS", 10)
RF_MAX_FEATURES = system.getenv_text("RF_MAX_FEATURES", "auto")

# Miscellaneous classifier options
USE_FEATURE_SELECTION = system.getenv_bool("USE_FEATURE_SELECTION", False)
ANOVA_KBEST = system.getenv_integer("ANOVA_KBEST", 5)

#-------------------------------------------------------------------------------

def sklearn_report(actual, predicted, labels, stream=sys.stdout):
    """Print classification analysis report for ACTUAL vs. PREDICTED indices with original LABELS and using STREAM"""
    stream.write("Performance metrics:\n")
    stream.write(metrics.classification_report(actual, predicted, target_names=labels))
    stream.write("Confusion matrix:\n")
    # TODO: make showing all cases optional
    possible_indices = range(len(labels))
    confusion = metrics.confusion_matrix(actual, predicted, possible_indices)
    # TODO: make sure not clipped
    stream.write("{cm}\n".format(cm=confusion))
    debug.trace_object(6, confusion, "confusion")
    return


def create_tabular_file(filename, data):
    """Create tabular FILENAME with SkLearn DATA for use with read_classification_data"""
    # Note: intended for comparing results here against tutorial (e.g., interactive use with iPython)
    with open(filename, "w") as f:
        for features in data:
            f.write(FIELD_SEP.join(features) + "\n")
    return


def safe_float(text, default=0):
    """Convert TEXT to number or DEFAULT"""
    value = default
    try:
        value = float(text)
    except ValueError:
        pass
    debug.trace_fmtd(10, "safe_float({t}, [{d}]) => {v}",
                     t=text, d=default, v=value)
    return value


def read_classification_data(filename, values_hashes=None):
    """Reads table with (non-unique) label and tab-separated value, encoding
    categorical or scalar values with unique integer
    Note: label made lowercase; result returned as tuple (headers, classes, features)"""
    # TODO: convert into enumeration; rework via csv-reader in standard lib
    # TODO: rename labels to classes and values to features
    debug.trace_fmtd(4, "read_classification_data({f})", f=filename)
    headers = []
    labels = []
    values = []
    num_features = 0
    if values_hashes is None:
        values_hashes = []
    
    with open(filename) as f:
        for (n, line) in enumerate(f):
            # Split line into feature values, making not of first set as headers.
            line = system.from_utf8(line).rstrip("\n")
            debug.trace_fmtd(6, "L{n}: {l}", n=(n + 1), l=line)
            if not line:
                debug.trace_fmtd(2, "Warning: Ignoring blank line at {f}:{n}",
                                 f=filename, n=(n + 1))
                continue
            items = line.split(FIELD_SEP)
            if (n == 0):
                num_features = len(items)
                headers = items
                continue
            debug.assertion(len(items) == num_features)

            # Make notes of scalar or categorical data
            # Note: Done over entire vector to map class value if needed;
            # and, 'NULL' gets mapped to None.
            for (i, v) in enumerate(items):
                if (i >= len(values_hashes)):
                    values_hashes.append({})
                ## BAD: num_hashed = len(values_hashes[i])
                new_hashed_value = False
                if v.lower() == "null":
                    items[i] = None
                    continue
                float_value = safe_float(v, None)
                if not float_value:
                    # Replace non-numeric value with index
                    if v not in values_hashes[i]:
                        values_hashes[i][v] = len(values_hashes[i])
                        ## BAD: items[i] = values_hashes[i].keys().index(v)
                        new_hashed_value = True
                    items[i] = values_hashes[i][v]
                ## BAD: if (num_hashed < len(values_hashes[i])):
                if new_hashed_value:
                    debug.trace_values(5, values_hashes[i], "values_hashes[{i}]".format(i=i))

            # Add class label and feature vector values
            if CLASS_FIRST:
                labels.append(items[0])
                values.append(items[1:])
            else:
                labels.append(items[num_features - 1])
                values.append(items[:num_features - 1])
            
    debug.trace_fmtd(5, "read_classification_data() => (h={h}, _, _, _)", h=headers)
    ## BAD: debug.trace_values(7, zip(labels, values), "zip(l, v) =>")
    ## BAD: debug.trace_object(7, "zip(l, v) => {z}", z=zip(labels, values))
    ## BAD: debug.trace_object(7, zip(labels, values), "zip(l, v) =>")
    debug.trace_fmtd(7, "zip(l, v) => {z}", z=list(zip(labels, values)))
    return (headers, labels, values, values_hashes)

#-------------------------------------------------------------------------------

class SupervisedClassifier(object):
    """Class for building text classification.
    Notes: This uses a pipeline approach for optional feature selection"""
    # NOTE: Anova filter based on example for Pipeline:
    #  https://scikit-learn.org/0.15/modules/generated/sklearn.pipeline.Pipeline.html
    # TODO: use subclasses rather than flags; add cross-fold validation support
    opt_anova_filter = SelectKBest(f_regression, k=ANOVA_KBEST)
    opt_feature_selection = ([] if not USE_FEATURE_SELECTION else
                             [('anova', opt_anova_filter)])
    cat_pipeline = Pipeline(opt_feature_selection + [('clf', MultinomialNB())])

    def __init__(self):
        """Class constructor"""
        debug.trace_fmtd(4, "sc.__init__(); self=={s}", s=self)
        self.keys = []
        self.classifier = None
        self.values_hash = None

        if USE_SVM:
            self.cat_pipeline = Pipeline(
                self.opt_feature_selection +
                [('clf', SVC(kernel=SVM_KERNEL,
                             C=SVM_PENALTY,
                             max_iter=SVM_MAX_ITER,
                             verbose=SVM_VERBOSE))])
        if USE_SGD:
            self.cat_pipeline = Pipeline(
                self.opt_feature_selection +
                ['clf', SGDClassifier(loss=SGD_LOSS,
                                      penalty=SGD_PENALTY,
                                      alpha=SGD_ALPHA,
                                      random_state=SGD_SEED,
                                      ## TODO: max_iter=SGD_MAX_ITER,
                                      n_iter=SGD_MAX_ITER,
                                      ## tol=SGD_TOLERANCE
                                      verbose=SGD_VERBOSE)])
        if USE_LR:
            self.cat_pipeline = Pipeline(
                self.opt_feature_selection +
                [('clf', LogisticRegression(penalty=LR_PENALTY))])

        if USE_RF:
            self.cat_pipeline = Pipeline(
                self.opt_feature_selection +
                [('clf', RandomForestClassifier(n_estimators=RF_NUM_ESTIMATORS,
                                                max_features=RF_MAX_FEATURES))])

        return

    def train(self, filename):
        """Train classifier using tabular FILENAME with label and text
        Note: Does on the fly conversion of non-numeric values to integers,
        recording mapping for use in test data mapping."""
        debug.trace_fmtd(4, "sc.train({f})", f=filename)
        (headers, labels, values, self.values_hash) = read_classification_data(filename)
        value_matrix = numpy.array(values, order='F')
        self.keys = sorted(numpy.unique(labels))
        ## TODO: debug.assertion(len(system.intersection(headers, self.keys)) == len(headers))
        label_indices = [self.keys.index(l) for l in labels]
        ## BAD: self.classifier = self.cat_pipeline.fit(values, label_indices)
        label_indice_vector = numpy.array(label_indices)
        self.classifier = self.cat_pipeline.fit(value_matrix, label_indice_vector)
        debug.trace_object(7, self.classifier, "classifier")
        return

    def test(self, filename, report=False, stream=sys.stdout):
        """Test classifier over tabular data from FILENAME with label and text, returning accuracy. 
        Notes:
        - Uses non-numeric value mapping created during training for mapping test values.
        - Optionally, a detailed performance REPORT is output to STREAM."""
        # TODO: convert list enumeration to interator-based reading
        debug.trace_fmtd(4, "sc.test({f})", f=filename)
        (all_headers, all_labels, all_values, self.values_hash) = read_classification_data(filename, self.values_hash)
        ## TODO: debug.assertion(len(system.intersection(all_headers, self.keys)) == len(all_headers))

        # TODO: use hash of positions
        actual_indices = []
        values = []
        labels = []
        for (i, label) in enumerate(all_labels):
            if label in self.keys:
                values.append(all_values[i])
                actual_indices.append(self.keys.index(label))
                labels.append(label)
            else:
                debug.trace_fmtd(1, "Ignoring test label {l} not in training data (line {n})",
                                 l=label, n=(i + 1))
        predicted_indices = self.classifier.predict(values)
        ## TODO: predicted_labels = [self.keys[i] for i in predicted_indices]
        num_ok = sum([(actual_indices[i] == predicted_indices[i]) for i in range(len(actual_indices))])
        accuracy = float(num_ok) / len(values)
        if report:
            if VERBOSE:
                stream.write("\n")
                stream.write("Actual\tPredict\n")
                for i in range(len(actual_indices)):
                    stream.write("{act}\t{pred}\n".
                                 format(act=self.keys[actual_indices[i]],
                                        pred=self.keys[predicted_indices[i]]))
                stream.write("\n")
            keys = self.keys
            sklearn_report(actual_indices, predicted_indices, keys, stream)
        if OUTPUT_BAD:
            bad_instances = "Actual\tBad\tFeatures\n"
            # TODO: for (i, actual_index) in enumerate(actual_indices)
            for i in range(len(actual_indices)):
                if (actual_indices[i] != predicted_indices[i]):
                    # TODO: why is pylint flagging the format string as invalid?
                    bad_instances += u"{g}\t{b}\t{f}".format(
                        g=self.keys[actual_indices[i]],
                        b=self.keys[predicted_indices[i]],
                        f=values)
            system.write_file(filename + ".bad", bad_instances)
        return accuracy

    def categorize(self, text):
        """Return category for TEXT"""
        # TODO: Add support for category distribution
        debug.trace_fmtd(4, "sc.categorize({_})")
        debug.trace_fmtd(6, "\ttext={t}", t=text)
        index = self.classifier.predict([text])[0]
        label = self.keys[index]
        debug.trace_fmtd(5, "categorize() => {r}", r=label)
        return label

    def save(self, filename):
        """Save classifier to FILENAME"""
        debug.trace_fmtd(4, "sc.save({f})", f=filename)
        system.save_object(filename, [self.keys, self.classifier])
        return

    def load(self, filename):
        """Load classifier from FILENAME"""
        debug.trace_fmtd(4, "sc.load({f})", f=filename)
        try:
            (self.keys, self.classifier) = system.load_object(filename)
        except (TypeError, ValueError):
            system.print_stderr("Problem loading classifier from {f}: {exc}".
                                format(f=filename, exc=sys.exc_info()))
        return

#------------------------------------------------------------------------

def main(args):
    """Supporting code for command-line processing"""
    print_stderr("Not intended for command-line usage: see SupervisedClassifier.")
    debug.assertion(len(args) == 1)
    return

if __name__ == '__main__':
    main(sys.argv)
