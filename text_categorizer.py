#! /usr/bin/env python
#
# Class for text categorizer using Scikit-Learn. See tutorial at
#    http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#
# TODO:
# - Maintain cache of categorization results.
# - Review categorization code and add examples for clarification of parameters.
#- - Fix SHOW_REPORT option for training.
#
# Copyright (c) 2017-2018 Scrappycito, LLC
#

"""Text categorization support"""

# Standard packages
import json
import os
import re
import sys
from collections import defaultdict

# Installed packages
import cherrypy
import numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Local packages
import debug
import system

SERVER_PORT = system.getenv_integer("SERVER_PORT", 9440)
OUTPUT_BAD = system.getenv_bool("OUTPUT_BAD", False)
CONTEXT_LEN = system.getenv_int("CONTEXT_LEN", 512)
VERBOSE = system.getenv_bool("VERBOSE", False)

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
    """Create tabular FILENAME with SkLearn DATA for use with read_categorization_data"""
    # Note: intended for comparing results here against tutorial
    with open(filename, "w") as f:
        for i in range(len(data.data)):
            text = system.to_utf8(re.sub("[\t\n]", " ", data.data[i]))
            f.write("{lbl}\t{txt}\n".format(lbl=data.target_names[data.target[i]], txt=text))
    return


def read_categorization_data(filename):
    """Reads table with (non-unique) label and tab-separated value. 
    Note: label made lowercase; result returned as tuple (labels, values)"""
    debug.trace_fmtd(4, "read_categorization_data({f})", f=filename)
    labels = []
    values = []
    with open(filename) as f:
        for (i, line) in enumerate(f):
            line = system.from_utf8(line)
            items = line.split("\t")
            if len(items) == 2:
                labels.append(items[0].lower())
                values.append(items[1])
            else:
                debug.trace_fmtd(4, "Warning: Ignoring item w/ unexpected format at line {num}",
                                 num=(i + 1))
    ## OLD: debug.trace_fmtd(7, "table={t}", t=table)
    debug.trace_values(7, zip(labels, values), "table")
    return (labels, values)


class TextCategorizer(object):
    """Class for building text categorization"""
    # TODO: add cross-fold validation support; make TF/IDF weighting optional
    cat_pipeline = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())])

    def __init__(self):
        """Class constructor"""
        debug.trace_fmtd(4, "tc.__init__(); self=={s}", s=self)
        self.keys = []
        self.classifier = None
        if USE_SVM:
            self.cat_pipeline = Pipeline(
                [('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', SVC(kernel=SVM_KERNEL,
                             C=SVM_PENALTY,
                             max_iter=SVM_MAX_ITER,
                             verbose=SVM_VERBOSE))])
        if USE_SGD:
            self.cat_pipeline = Pipeline(
                [('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(loss=SGD_LOSS,
                                       penalty=SGD_PENALTY,
                                       alpha=SGD_ALPHA,
                                       random_state=SGD_SEED,
                                       ## TODO: max_iter=SGD_MAX_ITER,
                                       n_iter=SGD_MAX_ITER,
                                       ## tol=SGD_TOLERANCE
                                       verbose=SGD_VERBOSE))])

        return

    def train(self, filename):
        """Train classifier using tabular FILENAME with label and text"""
        debug.trace_fmtd(4, "tc.train({f})", f=filename)
        (labels, values) = read_categorization_data(filename)
        self.keys = sorted(numpy.unique(labels))
        label_indices = [self.keys.index(l) for l in labels]
        self.classifier = self.cat_pipeline.fit(values, label_indices)
        debug.trace_object(7, self.classifier, "classifier")
        return

    def test(self, filename, report=False, stream=sys.stdout):
        """Test classifier over tabular data from FILENAME with label and text, returning accuracy. Optionally, a detailed performance REPORT is output to STREAM."""
        debug.trace_fmtd(4, "tc.test({f})", f=filename)
        ## OLD: (labels, values) = read_categorization_data(filename)
        (all_labels, all_values) = read_categorization_data(filename)

        ## BAD: actual_indices = [self.keys.index(l) for l in labels]
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
                debug.trace_fmtd(4, "Ignoring test label {l} not in training data (line {n})",
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
            ## BAD: sklearn_report(actual_indices, predicted_indices, self.keys, stream)
            ## OLD: keys = sorted(numpy.unique(labels))
            keys = self.keys
            sklearn_report(actual_indices, predicted_indices, keys, stream)
        if OUTPUT_BAD:
            bad_instances = "Actual\tBad\tText\n"
            # TODO: for (i, actual_index) in enumerate(actual_indices)
            for i in range(len(actual_indices)):
                if (actual_indices[i] != predicted_indices[i]):
                    text = values[i]
                    context = (text[:CONTEXT_LEN] + "...\n") if (len(text) > CONTEXT_LEN) else text
                    # TODO: why is pylint flagging the format string as invalid?
                    bad_instances += u"{g}\t{b}\t{t}".format(
                        g=self.keys[actual_indices[i]],
                        b=self.keys[predicted_indices[i]],
                        t=context)
            system.write_file(filename + ".bad", bad_instances)
        return accuracy

    def categorize(self, text):
        """Return category for TEXT"""
        # TODO: Add support for category distribution
        debug.trace_fmtd(4, "tc.categorize({_})")
        debug.trace_fmtd(6, "\ttext={t}", t=text)
        index = self.classifier.predict([text])[0]
        label = self.keys[index]
        debug.trace_fmtd(5, "categorize() => {r}", r=label)
        return label

    def save(self, filename):
        """Save classifier to FILENAME"""
        debug.trace_fmtd(4, "tc.save({f})", f=filename)
        system.save_object(filename, [self.keys, self.classifier])
        return

    def load(self, filename):
        """Load classifier from FILENAME"""
        debug.trace_fmtd(4, "tc.load({f})", f=filename)
        try:
            (self.keys, self.classifier) = system.load_object(filename)
        except (TypeError, ValueError):
            system.print_stderr("Problem loading classifier from {f}: {exc}".
                                format(f=filename, exc=sys.exc_info()))
        return

#-------------------------------------------------------------------------------
# CherryPy Web server based on following tutorial
#     https://simpletutorials.com/c/2165/How%20to%20Create%20a%20Simple%20JSON%20Service%20with%20CherryPy
#
# TODO: move to ~/visual-diff (e.g., text_categorizer_server.py)
#

INDEX_HTML = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <title>Text categorizer</title>
    </head>
    <body>
        Try <a href="categorize">categorize</a> and <a href="get_category_image">get_category_image</a>.<br>
        <p>
        note: You need to supply the <i>text</i> parameter. For example,
        <ul>
            <li>Category for <a href="categorize?text={donald_trump}">{donald_trump}</a>.</li>
            <li>Image for <a href="get_category_image?text={my_dog}">{my_dog}</a></li>
        </ul>
        <p>
        Other example(s):
        <ul>
            <li><a href="shutdown">Shutdown the server</a></li> 
        </ul>
    </body>
</html>
""".format(donald_trump="Donald Trump is President.",
           my_dog="My dog has fleas.")
# TODO: only show shutdown for debugging hosts

CATEGORY_IMAGE_HASH = {
    # TODO: just use <category>.png to eliminate the hash
    # NOTES:
    # - drugs conflates with health
    # - government conflated with politics
    # - pets conflated with animal
    "animal": "/static/animals.png",
    "art": "/static/art.png",
    "business": "/static/business.jpg",
    "computers": "/static/computers.jpg",
    "drugs": "/static/health.jpg",
    "econimics": "/static/econimics.jpg",
    "education": "/static/education.png",
    "engineering": "/static/engineering.jpg",
    "food": "/static/food.jpg",
    "geography": "/static/geography.png",
    "geometry": "/static/geometry.jpg",
    "government": "/static/politics.png",
    "health": "/static/health.jpg",
    "history": "/static/history.jpg",
    "internet": "/static/internet.jpg",
    "law": "/static/law.jpg",
    "mathematics": "/static/mathematics.jpg",
    "military": "/static/military.png",
    "movie": "/static/movie.jpg",
    "music": "/static/music.jpg",
    "pets": "/static/animals.png",
    "philosophy": "/static/philosophy.jpg",
    "politics": "/static/politics.png",
    "psychology": "/static/psychology.png",
    "religion": "/static/religion.jpg",
    "science": "/static/science.png",
    "software": "/static/software.jpg",
    "sports": "/static/sports.jpg",
    "technology": "/static/technology.jpg",
    "television": "/static/television.jpg",
    "tools": "/static/tools.jpg",
    "weather": "/static/weather.png",
}


class web_controller(object):
    """Controller for CherryPy web server with embedded text categorizer"""
    # TODO: put visual-diff support in ~/visual-diff directory (e.g., category image mapping)
    
    def __init__(self, model_filename, *args, **kwargs):
        """Class constructor: initializes search engine server"""
        debug.trace_fmtd(5, "web_controller.__init__(s:{s}, a:{a}, kw:{k})__",
                         s=self, a=args, k=kwargs)
        self.text_cat = TextCategorizer()
        self.text_cat.load(model_filename)
        self.category_image = defaultdict(lambda: "/static/unknown-with-question-marks.png")
        # HACK: wikipedia categorization specific
        self.category_image.update(CATEGORY_IMAGE_HASH)
        # Note: To avoid cross-origin type errrors, Access-Control-Allow-Origin
        # is made open. See following:
        # - http://cleanbugs.com/item/how-to-get-cross-origin-sharing-cors-post-request-working-a-resource-413656.html
        # - https://stackoverflow.com/questions/6054473/python-cherrypy-how-to-add-header
        # TODO: put cherrypy config in start_web_controller (or put it's configuration here)
        ## BAD: cherrypy.response.headers["Access-Control-Allow-Origin"] = "*"
        return

    @cherrypy.expose
    def index(self, **kwargs):
        """Website root page (e.g., web site overview and link to search)"""
        debug.trace_fmtd(6, "wc.index(s:{s}, kw:{kw})", s=self, kw=kwargs)
        ## OLD: return "not much here excepting categorize and get_category_image"
        return (INDEX_HTML)

    @cherrypy.expose
    def categorize(self, text, **kwargs):
        """Infer category for TEXT"""
        debug.trace_fmtd(6, "wc.categorize(s:{s}, _, kw:{kw})", s=self, kw=kwargs)
        return self.text_cat.categorize(text)

    @cherrypy.expose
    ## @cherrypy.tools.json_out()
    def get_category_image(self, text, **kwargs):
        """Infer category for TEXT and return image"""
        debug.trace_fmtd(5, "wc.get_category_image(_, {kw}); self={s}", t=text, s=self, kw=kwargs)
        cat = self.categorize(text, **kwargs)
        image = self.category_image[cat]
        # for JSONP, need to add callback call and format the call
        # TODO: see if cherrypy handles this
        # see https://stackoverflow.com/questions/19456146/ajax-call-and-clean-json-but-syntax-error-missing-before-statement
        ## return image
        ## return json.dumps({"image": image})
        ## return {"image": image}
        image_id = kwargs.get("id", "id0")
        result = json.dumps({"image": image, "id": image_id})
        if 'callback' in kwargs:
            callback_function = kwargs['callback']
            data = kwargs.get("data", "")
            result = (callback_function + "(" + result + ", " + data + ");")
        debug.trace_fmtd(6, "wc.get_category_image() => {r}", r=result)
        return result

    @cherrypy.expose
    def stop(self, **kwargs):
        """Stops the web search server and saves cached data to disk"""
        debug.trace_fmtd(5, "wc.stop(s:{s}, kw:{kw})", s=self, kw=kwargs)
        if os.environ.get("HOST_NICKNAME") in ["hostwinds", "ec2-micro"]:
            return "Call security!"
        cherrypy.engine.stop()
        cherrypy.engine.exit()
        # TODO: use HTML so shutdown shown in title
        return "Adios"

    # alias for stop
    shutdown = stop
    # TODO: track down delay in python process termination


def start_web_controller(model_filename):
    """Start up the CherryPy controller for categorization via MODEL_FILENAME"""
    # TODO: return status code
    debug.trace(5, "start_web_controller()")

    # Load in CherryPy configuration
    # TODO: use external configuration file
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd()),
            ## take 2: on avoiding cross-origin type errrors
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [
                ## OLD: ('Content-Type', 'text/javascript'),
                ('Access-Control-Allow-Origin', '*'),
            ]
        },
        'global': {
            'server.socket_host': "0.0.0.0",
            'server.socket_port': SERVER_PORT,
            'server.thread_pool': 10,
            }
        }

    # Start the server
    # TODO: trace out all configuration settings
    debug.trace_values(4, cherrypy.response.headers, "default response headers")
    cherrypy.quickstart(web_controller(model_filename), "", conf)
    ## TODO: debug.trace_value(4, cherrypy.response.headers, "response headers")
    cherrypy.engine.start()
    return


#------------------------------------------------------------------------

def main(args):
    """Supporting code for command-line processing"""
    debug.trace_fmtd(6, "main({a})", a=args)
    if (len(args) != 2):
        system.print_stderr("Usage: {p} model".format(p=args[0]))
        return
    model = args[1]
    start_web_controller(model)
    return

if __name__ == '__main__':
    main(sys.argv)
