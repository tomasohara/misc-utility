#! /usr/bin/env python
#
# Illustrates how to use pandas and sklearn to do machine learning. This
# was initially over Iris dataset, based on following:
#    https://medium.com/codebagng/basic-analysis-of-the-iris-data-set-using-python-2995618a6342.
#
# Notes:
# - Environment variables:
#   DATA_FILE  FIELD_SEP  SCORING_METRIC  SEED  SKIP_DEVEL  SKIP_PLOTS  USE_DATAFRAME  VALIDATE_ALL  VALIDATION_CLASSIFIER  VERBOSE 
# - Currently only supports cross-validation (i.e., partitions of single datafile).
# - This partititions training data into development and validation sets.
# - Also does k-fold cross validation over development data split using 1/k-th as test.
# - That is, the training data is partitioned twice.
# - As an expediency to disable validation, epsilon is used for validation percent (e.g., 1e-6), because sklearn doesn't allow the percent to be specified as zero.
#
# TODO:
# - Add exception handling throughout.
# 

"""Illustrates sklearn classification over data with panda csv-based import"""

# Standard packages
import sys

# Installed pckages
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
## TEST:
import tensorflow as tf

# Local packages
import debug
import system
import text_utils
from text_utils import version_to_number
import tf_dnn

# note: Unfortunately, tensorflow dropped contrib in version 2.0.
V1_13_1 = version_to_number("1.13.1")
V2_0 = version_to_number("2.0")
debug.assertion(V1_13_1 <= version_to_number(tf.version.VERSION) < V2_0)

#................................................................................
# Constants (e.g., environment-based options)

VERBOSE = system.getenv_bool("VERBOSE", False)
DATA_FILE = system.getenv_text("DATA_FILE", "iris.csv")
FIELD_SEP = system.getenv_text("FIELD_SEP", ",")
USE_DATAFRAME = system.getenv_bool("USE_DATAFRAME", False)
IGNORE_FIELDS = text_utils.extract_string_list(system.getenv_text("IGNORE_FIELDS", ""))
SKIP_PLOTS = system.getenv_bool("SKIP_PLOTS", False)
OUTPUT_CSV = system.getenv_bool("OUTPUT_CSV", False)
SKIP_VALIDATION = system.getenv_bool("SKIP_VALIDATION", False)
TENSOR_FLOW_CLASSIFIER = "TF-dnn"        # TF Dense Neural Network (see tf_dnn.py)
KNN_CLASSIFIER = "KNN"
DEFAULT_DEVEL_CLASSIFIER = None if (not SKIP_VALIDATION) else TENSOR_FLOW_CLASSIFIER
DEVEL_CLASSIFIER = system.getenv_text("DEVEL_CLASSIFIER", DEFAULT_DEVEL_CLASSIFIER)
EPSILON = 1.0e-6
DEFAULT_VALIDATION_PCT = 0.20 if (not SKIP_VALIDATION) else EPSILON
VALIDATION_PCT = system.getenv_number("VALIDATION_PCT", DEFAULT_VALIDATION_PCT)
VALIDATE_ALL = system.getenv_bool("VALIDATE_ALL", False)
DEFAULT_VALIDATION_CLASSIFIER = KNN_CLASSIFIER if (not SKIP_VALIDATION) else None
VALIDATION_CLASSIFIER = system.getenv_text("VALIDATION_CLASSIFIER", DEFAULT_VALIDATION_CLASSIFIER)
SKIP_DEVEL = system.getenv_bool("SKIP_DEVEL", False)
TEST_PCT = system.getenv_number("TEST_PCT", 0.10)
SEED = system.getenv_bool("SEED", 7919)
SCORING_METRIC = system.getenv_text("SCORING_METRIC", "accuracy")

#...............................................................................
# Utility functions

def create_feature_mapping(label_values):
    """Return hash mapping elements from LABEL_VALUES into integers"""
    # EX: create_feature_mapping(['c', 'b, 'b', 'a']) => {'c':0, 'b':1, 'a':2}
    debug.assertion(isinstance(label_values, list))
    id_hash = {}
    for item in label_values:
        if (item not in id_hash):
            id_hash[item] = len(id_hash)
    debug.trace_fmtd(7, "create_feature_mapping({l}) => {h}", l=label_values, h=id_hash)
    return id_hash

#...............................................................................
# Main processing

def main():
    """Entry point for script"""
    # Read the data
    dataset = pandas.read_csv(DATA_FILE, sep=FIELD_SEP)
    feature_names = list(dataset.columns[0:-1])
    class_var = dataset.columns[-1]
    debug.trace_fmtd(4, "class_var={c} features:{f}",
                     f=feature_names, c=class_var)
    debug.trace_object(7, dataset, "dataset")

    # Optionally remove specified fields from data
    if IGNORE_FIELDS:
        debug.assertion(not system.difference(IGNORE_FIELDS, feature_names))
        dataset = dataset.drop(IGNORE_FIELDS, axis=1)
        feature_names = list(dataset.columns[0:-1])

    # Show samples from the data along summary statistics and other information
    try:
        # Show first 10 rows
        print("Sample of data set (head, tail, random):")
        print(dataset.head())
        # SHow last 10 row
        print(dataset.tail())
        # Show 5 random rows
        print(dataset.sample(5))
        # Show a statistical summary about the dataset.
        print("statistical summary:")
        print(dataset.describe())
        # Show how many null entries are in the dataset.
        print("Null count:")
        print(dataset.isnull().sum())
    except:
        debug.trace_fmtd(2, "Error: Problem during dataset illustration, etc.: {exc}",
                         exc=sys.exc_info())
        debug.raise_exception(6)
    
    # Optionally show some plot
    if (not SKIP_PLOTS):
            
        # box and whisker plots
        dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
        plt.show()
        
        # histograms
        dataset.hist()
        plt.show()
        
        # scatter plot matrix
        scatter_matrix(dataset)
        plt.show()

    # Split-out validation dataset
    if USE_DATAFRAME:
        features_indices = dataset.columns
        num_features = len(features_indices) - 1
        X = dataset[features_indices[0:num_features]]
        y = dataset[features_indices[num_features]]
    else:
        array = dataset.values
        debug.trace_object(7, array, "array")
        num_features = (array.shape[1] - 1)
        X = array[:, 0:num_features]
        y = array[:, num_features]
    debug.trace_fmtd(7, "X={X}\ny={y}", X=X, y=y)
    if OUTPUT_CSV:
        # TODO: drop pandas index column (first one; no header)
        basename = system.remove_extension(DATA_FILE)
        X.to_csv(basename + "-X.csv", sep=FIELD_SEP)
        y.to_csv(basename + "-y.csv", sep=FIELD_SEP)
    X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=VALIDATION_PCT, random_state=SEED)
    ## TODO:
    debug.trace_fmtd(6, "X_train={xt}\nX_valid={xv}\ny_train={yt}\ny_valid={yv}", xt=X_train, xv=X_validation, yt=y_train, yv=y_validation)
    
    # Test options and evaluation metric
    ## NOTE: precision and recall currently not supported (see below)
    y_values = list(y.values) if USE_DATAFRAME else y
    num_classes = len(create_feature_mapping(y_values))
    is_binary = (num_classes == 2)
    
    # Spot check algorithms
    # TODO: Add legend for non-standard abbreviations.
    models = []
    models.append(("LR", LogisticRegression()))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append((KNN_CLASSIFIER, KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))        #  Classification and Regression Trees
    models.append(("GNB", GaussianNB()))
    models.append(("MNB", MultinomialNB()))
    models.append(("SVM", SVC()))
    ## BAD: models.append(("TF", tf_dnn.NeuralNetworkClassifier()))
    ## TEST:
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)      # pylint: disable=no-member
    models.append((TENSOR_FLOW_CLASSIFIER, tf_dnn.NeuralNetworkClassifier(feature_columns=feature_columns)))

    # Evaluate each model in turn.
    # TODO: show precision, recall, F1, as well as accuracy
    # Sample results:
    # name  acc    stdev
    # LR    0.967  0.041
    # LDA   0.975  0.038
    # KNN   0.983  0.033
    # CART  0.975  0.038
    # GNB   0.975  0.053
    # SVM   0.982  0.025
    #    
    summaries = []
    if (not SKIP_DEVEL):
        print("Sample development test set results using scoring method {sm}".format(sm=SCORING_METRIC))
        ## TODO: average = "micro" if (not is_binary) else None
        for name, model in models:
            if (name != DEVEL_CLASSIFIER):
                continue
            kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=SEED)
            ## TODO: cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=SCORING_METRIC, average=average)
            ## TODO: get this to work when SCORING_METRIC is not accuracy (which leads to not supported error for multicalss data)
            ## (e.g., add environment variable so that sklearn uses micro or macro average
            try:
                cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=SCORING_METRIC)
                summaries.append("{n}\t{avg}\t{std}".format(n=name, avg=system.round_num(cv_results.mean()), std=system.round_num(cv_results.std())))
                ## OLD: model.fit(X_train, y_train)
                ## OLD: debug.trace_fmtd(4, "training data score: {s}", s=model.score(X_train, y_train))
                # Show confusion matrix for sample split of training data
                if VERBOSE:
                    X_devel, X_test, y_devel, y_test = model_selection.train_test_split(X_train, y_train, test_size=TEST_PCT, random_state=SEED)
                    model.fit(X_devel, y_devel)
                    debug.trace_fmtd(4, "devel data score: {s}", s=model.score(X_devel, y_devel))
                    model.fit(X_test, y_test)
                    debug.trace_fmtd(4, "test data score: {s}", s=model.score(X_test, y_test))
                    predictions = model.predict(X_test)
                    print("Development test set confusion matrix:")
                    print(confusion_matrix(y_test, predictions))
                    print("Development test classification report:")
                    print(classification_report(y_test, predictions))
            except:
                debug.trace_fmtd(2, "Error: Problem during training evaluation: {exc}",
                                 exc=sys.exc_info())
                debug.raise_exception(6)
        print("Cross validation results over development test set")
        print("name\tacc\tstdev")
        print("\n".join(summaries))
    
    # Make predictions on validation dataset
    # Sample output:
    # Confusion matrix:
    # [[ 7  0 0]
    #  [ 0 11 1]
    #  [ 0  2 9]]
    # classification report:
    #              precision recall  f1  support
    # Iris-setosa     1.00   1.00  1.00  7
    # Iris-versicolor 0.85   0.92  0.88  12
    # Iris-virginica  0.90   0.82  0.86  11
    # avg / total     0.90   0.90  0.90  30
    # TODO: rework so that loop bypassed if no validation set
    average = "micro" if (not is_binary) else "binary"
    num_run = 0
    for name, model in models:
        if (not (VALIDATE_ALL or (name == VALIDATION_CLASSIFIER))):
            continue
        try:
            if VALIDATE_ALL:
                print("." * 80)
            print("Resuts over validation data for {n}:".format(n=name))
            num_run += 1
            model.fit(X_train, y_train)
            debug.trace_fmtd(4, "training data score: {s}", s=model.score(X_train, y_train))
            predictions = model.predict(X_validation)
            print("confusion matrix:")
            print(confusion_matrix(y_validation, predictions))
            ## TODO: drop accurracy ... F1 (provide in report)
            if VERBOSE:
                print("accuracy:")
                print(accuracy_score(y_validation, predictions))
                print("precision:")
                print(precision_score(y_validation, predictions, average=average))
                print("recall:")
                print(recall_score(y_validation, predictions, average=average))
                print("F1:")
                print(f1_score(y_validation, predictions, average=average))
            print("classification report:")
            print(classification_report(y_validation, predictions))
        except:
            debug.trace_fmtd(2, "Error: Problem during validation evaluation: {exc}",
                             exc=sys.exc_info())
            debug.raise_exception(6)
    if ((num_run == 0) and VALIDATION_CLASSIFIER):
        system.print_stderr("Error: Validation classifier '{clf}' not supported", clf=VALIDATION_CLASSIFIER)
    
#------------------------------------------------------------------------

if __name__ == "__main__":
    main()
