#! /usr/bin/env python
#
# Use scikit-learn to perform probabilistic grid search.
#
# This extends the following brute-force approach to optionally support probabilistic search:
#    https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# It is also based on the multiclass support from the following:
#    https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library
#
# TODO:
# - Parameterize the following deep learning options:
#   activation, ...
#

"""Keras for probabilistic grid search via scikit-learn"""

# Standard packages
import sys

# Installed packages
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy
import pandas
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# Local packages
import debug
import system
import text_utils

#...............................................................................
# Constants (e.g., based on environment)

DEFAULT_VERBOSITY = ((debug.get_level() + 1) // 2)
VERBOSITY_LEVEL = system.getenv_int("VERBOSITY_LEVEL", DEFAULT_VERBOSITY)
DATA_FILE = system.getenv_text("DATA_FILE", "samples/pima-indians-diabetes.csv")
FIELD_SEP = system.getenv_text("FIELD_SEP", ",")
BRUTE_FORCE = system.getenv_bool("BRUTE_FORCE", False)
RANDOM_OPTIMIZATION = (not BRUTE_FORCE)
SEED = system.getenv_bool("SEED", 7919)

# TODO: Add descriptions for important deep learning parameters (e.g., NUM_EPOCHS and BATCH_SIZE).
NUM_EPOCHS = system.getenv_int("NUM_EPOCHS", 200)
BATCH_SIZE = system.getenv_int("BATCH_SIZE", 5)
NUM_FOLDS = system.getenv_int("NUM_FOLDS", 10)
NUM_JOBS = system.getenv_int("NUM_JOBS", -1, "Number of parallel jobs (-1 uses cores)")
NUM_ITERS = system.getenv_int("NUM_ITERS", 100)
SCORING_METRIC = system.getenv_text("SCORING_METRIC", "accuracy")
USE_ONE_HOT = system.getenv_bool("USE_ONE_HOT", False)

QUICK_SEARCH = system.getenv_bool("QUICK_SEARCH", False)
NUM_EPOCH_VALUES = ([10, 50, 100, 250, 1000] if (not QUICK_SEARCH) else [10, 100, 1000])
BATCH_SIZE_VALUES = ([10, 20, 40, 60, 80, 100] if (not QUICK_SEARCH) else [10, 50, 100])

#...............................................................................
# Utility functions

def round3(num):
    """Round NUM using precision of 3"""
    return system.round_num(num, 3)

# TODO: put following in new ml_utils.py module
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
# Grid search support

def create_keras_model(num_input_features=None, num_classes=None):
    """Create n-layer Keras model with NUM_INPUT_FEATURES and NUM_CLASSED"""
    debug.trace_fmt(5, "create_keras_model(#f={nf}, #c={nc})",
                    nf=num_input_features, nc=num_classes)

    # Initialize defaults
    if (num_input_features is None):
        debug.trace(2, "Warning: number of features not specified so using 100!")
        num_input_features = 100
    if (num_classes is None):
        debug.trace(2, "Warning: number of classes not specified so using 2!")
        num_classes = 2
    is_binary = (num_classes == 2)

    # Create the model with optional hidden layers
    # TODO: parameterize activation fn; base number of hidden input on feature set size
    model = Sequential()
    model.add(Dense(12, input_dim=num_input_features, activation="relu"))
    # Add output layer
    if is_binary:
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(num_classes, activation="softmax"))

    # Compile model
    loss_function = "binary_crossentropy" if is_binary else "categorical_crossentropy"
    model.compile(loss=loss_function, optimizer="adam", metrics=[SCORING_METRIC])
    debug.trace_object(5, model, "model")
    debug.trace_fmt(4, "create_keras_model() => {m}", m=model)

    return model


#................................................................................

def main():
    """Main entry point for script"""
    # fix random seed for reproducibility
    numpy.random.seed(SEED)

    # load dataset
    # TODO: only skip first row if all symbolic
    headers = system.read_entire_file(DATA_FILE).split("\n")[0].split(FIELD_SEP)
    debug.assertion(all([text_utils.is_symbolic(v) for v in headers]))
    ## OLD: dataset = numpy.loadtxt(DATA_FILE, delimiter=FIELD_SEP, skiprows=1)
    data_frame = pandas.read_csv(DATA_FILE, sep=FIELD_SEP)
    dataset = data_frame.values
    
    # split into input (X) and output (y) variables
    num_features = (dataset.shape[1] - 1)
    debug.assertion(len(headers) == (num_features + 1))
    ## OLD: X = dataset[:, 0:num_features]
    X = dataset[:, 0:num_features].astype(float)
    y = dataset[:, num_features]
    ## TEST:
    y = list(y)
    debug.trace_fmtd(7, "X={X}\ny={y}", X=X, y=y)
    y_hash = create_feature_mapping(y)
    num_categories = len(y_hash)

    # Encode class values as integers, using one-hot vectors (i.e., one vector per category).
    symbolic_classes = all([text_utils.is_symbolic(v) for v in y])
    modified_y = y
    if symbolic_classes:
        encoder = LabelEncoder()
        encoder.fit(modified_y)
        modified_y = encoder.transform(y)
        debug.trace_fmtd(7, "encoded_y={ey}", ey=modified_y)
        
        # Convert integers to dummy variables (i.e,. one hot encoded)
        if USE_ONE_HOT:
            modified_y = np_utils.to_categorical(modified_y)
            debug.trace_fmtd(7, "one_hot_y={ohy}", ohy=modified_y)
        debug.trace_fmtd(8, "modified_y={my}", my=modified_y)

    # Create initial model
    create_model_fn = lambda: create_keras_model(num_input_features=num_features,
                                                 num_classes=num_categories)
    ## OLD: model = KerasClassifier(build_fn=create_model_fn, verbose=VERBOSITY_LEVEL)
    ## TODO: see why batch_size needed for better accuracy
    # Note: all grid-search parameters need to be specified in the classifier constructor call.
    model = KerasClassifier(build_fn=create_model_fn, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                            verbose=VERBOSITY_LEVEL)
    
    # Run standard classificaion
    # note: Used for comparison against samples/keras_multiclass.py.
    try:
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True)
        results = cross_val_score(model, X, modified_y, cv=kfold)
        print("{k}-fold cross validation results:".format(k=NUM_FOLDS))
        print("Baseline: mean={m} stdev={s}; num_epochs={ne} batch_size={bs}".format(
            m=round3(results.mean()), s=round3(results.std()), ne=NUM_EPOCHS, bs=BATCH_SIZE))
    except:
        debug.trace_fmtd(2, "Error: Problem during cross_val_score: {exc}", exc=sys.exc_info())
        debug.raise_exception(6)

    # Define the grid search parameters and then run the search with all cores and
    # 10-fold cross validation (by default).
    parameters = {"batch_size": BATCH_SIZE_VALUES,
                  "epochs": NUM_EPOCH_VALUES}
    debug.trace_fmt(4, "parameters: {p}", p=parameters)

    # Note: much better results with 10-fold cross validation vs. 3-fold
    try:
        if BRUTE_FORCE:
            grid = GridSearchCV(model, parameters, n_jobs=NUM_JOBS, cv=NUM_FOLDS, verbose=VERBOSITY_LEVEL)
        else:
            grid = RandomizedSearchCV(model, parameters, n_jobs=NUM_JOBS, n_iter=NUM_ITERS, cv=NUM_FOLDS, verbose=VERBOSITY_LEVEL)
        grid_result = grid.fit(X, modified_y)
        debug.trace_object(5, grid_result, "grid_result")
    except:
        debug.trace_fmtd(2, "Error: Problem during grid search: {exc}", exc=sys.exc_info())
        debug.raise_exception(6)
        
    # Summarize randomized grid search results
    try:
        gridsearch_type = "Randomized" if (not BRUTE_FORCE) else "Brute-force"
        print("{gt} gridsearch results:".format(gt=gridsearch_type))
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_["mean_test_score"]
        stds = grid_result.cv_results_["std_test_score"]
        params = grid_result.cv_results_["params"]
        print("Mean\tStdev\tParam")
        for mean, stdev, param in zip(means, stds, params):
            print("{m}\t{s}\t{p}".format(m=round3(mean), s=round3(stdev), p=param))
    except:
        debug.trace_fmtd(2, "Error: Problem during summarization: {exc}", exc=sys.exc_info())
        debug.raise_exception(6)
    
#...............................................................................

if __name__ == "__main__":
    main()
