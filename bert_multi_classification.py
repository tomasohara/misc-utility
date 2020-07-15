#! /usr/bin/env python
#
# Uses BERT for multiple-label classification, based on following blog:
#    https://towardsdatascience.com/beginners-guide-to-bert-for-multi-classification-task-92f5445c2d7c
#
# Notes:
# - Setup instructions:
#   -- sudo pip3 install virtualenv
#   -- virtualenv bertenv
#   -- python3 -m venv bertenv
#   -- source bertenv/bin/activate
#   -- tensorflow >= 1.11.0   # CPU Version of TensorFlow.
#   -- tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
# - Sample invocation:
#   TODO
#
# TODO:
# - Use Main class to add in support for argument parsing
# - Generalize to handling other data files.
# - Add options for BERT model and other parameters.
# - Convert into calling module for run_classifier.py directly (i.e., without using shell).
# - Rework sample invocation so that environment settings are temporary.
#

"""BERT for multiple-label classification"""

## TODO: import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
 
import tensorflow
# TODO: make the following optional (so that tensorflow_cpu can be used)
## OLD import tensorflow_gpu

import debug
from regex import my_re
import glue_helpers as gh
import system

# Note: the defaults are unintuitive, but they match the blog article.
MODEL_DIR = system.getenv_text("MODEL_DIR", "./model")
## TODO: BERT_DIR = system.getenv_text("BERT_DIR", MODEL_DIR)
DATA_DIR = system.getenv_text("DATA_DIR", "./dataset")
OUTPUT_DIR = system.getenv_text("OUTPUT_DIR", "./bert_output")
TASK_NAME = system.getenv_text("TASK_NAME", "cola")
USE_TSV_INPUT = system.getenv_bool("USE_TSV_INPUT", False)
CLASSIFIER_INVOCATION = system.getenv_text("CLASSIFIER_INVOCATION", "run_classifier.py")
# Note: USE_TSV_INPUT implies you use a tab-separated format properly formatted for BERT,
# so set BERT_FORMATTED False (e.g., USE_TSV_INPUT=1 BERT_FORMATTED=0 bert_multi_classification.py ...)
BERT_FORMATTED = system.getenv_bool("BERT_FORMATTED", USE_TSV_INPUT)
LOWER_CASE = system.getenv_bool("LOWER_CASE", False)
USE_ALBERT_DEFAULT = ("albert" in CLASSIFIER_INVOCATION)
USE_ALBERT = system.getenv_bool("USE_ALBERT", USE_ALBERT_DEFAULT)
BERT_NAME = "bert" if (not USE_ALBERT) else "albert"
CONFIG_FILE_DEFAULT = system.form_path(MODEL_DIR, "{b}_config.json".format(b=BERT_NAME))
BERT_CONFIG_FILE = system.getenv_text("BERT_CONFIG_FILE", CONFIG_FILE_DEFAULT)
                                     

# Get label list and split into columns
INPUT_LABELS = system.getenv_text("INPUT_LABELS", "id, label, text")
## TODO: INPUT_LABEL_LIST = re.split(r"\s, *", INPUT_LABELS)
# Labels to use for ID, CLASS_VALUE, and TEXT
INPUT_LABEL_LIST = re.split(r",", INPUT_LABELS)
debug.assertion(len(INPUT_LABEL_LIST) == 3)
ID_COL = INPUT_LABEL_LIST[0]
LABEL_COL = INPUT_LABEL_LIST[1]
TEXT_COL = INPUT_LABEL_LIST[2]

#-------------------------------------------------------------------------------
# General purpose helper functions
# TODO: put into text_utils, etc.

def version_to_number(version, max_padding=3):
    """Converts VERSION to number that can be used in comparisons
    Note: The Result will be of the form M.mmmrrrooo..., where M is the
    major number m is the minor, r is the revision and o is other.
    Each version component will be prepended with up MAX_PADDING [3] 0's
    Notes:
    - strings in the version are ignored
    - 0 is returned if version string is non-standard"""
    # EX: version_to_number("1.11.1") => 1.00010001
    # EX: version_to_number("1") => 1
    # EX: version_to_number("") => 0
    # TODO: support string (e.g., 1.11.2a).
    version_number = 0
    version_text = version
    new_version_text = ""
    max_component_length = (1 + max_padding)
    debug.trace_fmt(5, "version_to_number({v})", v=version)

    # Remove all alphabetic components
    version_text = re.sub(r"[a-z]", "", version_text, re.IGNORECASE)
    if (version_text != version):
        debug.trace_fmt(2, "Warning: stripped alphabetic components from version: {v} => {nv}", v=version, nv=version_text)

    # Remove all spaces (TODO: handle tabs and other whitespace)
    version_text = version_text.replace(" ", "")

    # Convert component numbers iteratively and zero-pad if necessary
    # NOTE: Components greater than max-padding + 1 treated as all 9's.
    debug.trace_fmt(4, "version_text: {vt}", vt=version_text)
    first = False
    num_components = 0
    regex = r"^(\d+)(\.((\d*).*))?$"
    while (my_re.search(regex, version_text)):
        component = my_re.group(1)
        # TODO: fix my_re.group to handle None as ""
        version_text = my_re.group(2) if my_re.group(2) else ""
        num_components += 1
        debug.trace_fmt(4, "new version_text: {vt}", vt=version_text)

        component = system.to_string(system.to_int(component))
        if first:
            new_version_text = component + "."
            regex = r"^(\d+)\.?((\d*).*)$"
        else:
            if (len(component) > max_component_length):
                old_component = component
                component = "9" * max_component_length
                debug.trace_fmt(2, "Warning: replaced overly long component #{n} {oc} with {c}",
                                n=num_components, oc=old_component, nc=component)
            new_version_text += component
            debug.trace_fmt(4, "Component {n}: {c}", n=num_components, c=component)
    version_number = system.to_float(new_version_text, version_number)
    ## TODO:
    ## if (my_re.search(p"[a-z]", version_text, re.IGNORECASE)) {
    ##     version_text = my_re.... 
    ## }
    debug.trace_fmt(4, "version_to_number({v}) => {n}", v=version, n=version_number)
    return version_number

#-------------------------------------------------------------------------------
# Helper functions specific to BERT

def ensure_bert_data_frame(data_frame, is_test=False):
    """Ensures data frame is in BERT format from input DATA_FRAME, using dummy values for alpha
    column.
    Notes:
    - See comments in blog mentioned in header.
    - Uses global costant BERT_FORMATTED."""
    # TODO: add parameter mapping input column names to ones assumed here (i.e., id, label, & text)
    debug.trace_fmt(5, "ensure_bert_data_frame({df})", df=data_frame)
    df_bert = None
    if BERT_FORMATTED:
        df_bert = data_frame
    else:
        try:
            first_sentence = data_frame[TEXT_COL][0]
            debug.trace_fmt(5, "First sentence: {s}", s=first_sentence)
            # Ignore header column if given (TODO, add parameter to make this optional)
            # Note: This assumes single word tokens without punctuation can't be a sentence.
            if re.search(r"^\w+$", first_sentence):
                debug.trace("Removing presumed header row of data frame")
                debug.assertion(is_test)
                data_frame = data_frame.drop(data.index[0])
                
            data_hash = {'guid': data_frame[ID_COL],
                         'alpha': (['-'] * data_frame.shape[0]),
                         'text': data_frame[TEXT_COL]}
            if not is_test:
                data_hash['label'] = data_frame[LABEL_COL]
                
            df_bert = pd.DataFrame(data_hash)
        except:
            debug.raise_exception(5)
            system.print_stderr("Exception converting data frame to BERT format: {exc}",
                                exc=sys.exc_info())
    debug.trace_fmt(4, "ensure_bert_data_frame(_) => {r}", r=df_bert)
    return df_bert

#--------------------------------------------------------------------------------

V1_11_0 = version_to_number("1.11.0")

debug.assertion(V1_11_0 <= version_to_number(tensorflow.__version__))
## OLD: debug.assertion(V1_11_0 <= version_to_number("tensorflow_gpu.__version__"))

#-------------------------------------------------------------------------------

## TODO: changes to BERT run_classifier.py code for different label sets
## NOTE: not needed if running one of GLUE tasks with support built into classifier (e.g., CoLA)
## def get_labels(self):
##     return ["0", "1"]
## def get_labels(self):
##    return ["0", "1", "2", "3", "4"]
## def get_labels(self):
##    return ["POSITIVE", "NEGATIVE"]

def main():
    """Entry point for script"""
    ## sudo apt-get install python3-pip
    ## a550d    1    a    To clarify, I didn't delete these pages.kcd12    0    a    Dear god this site is horrible.7379b    1    a    I think this is not appropriate.cccfd    2    a    The title is fine as it is.
    ## guid     textcasd4    I am not going to buy this useless stuff.3ndf9    I wanna be the very best, like no one ever was
    ## pip install pandas
    ## pip install sklearn
    ## id,text,labelsadcc,This is not what I want.,1cj1ne,He seriously have no idea what it is all about,0123nj,I don't think that we have any right to judge others,2
    in_seperator = ","
    in_ext = ".csv"
    if USE_TSV_INPUT:
        in_seperator = "\t"
        in_ext = ".tsv"
    df_train = pd.read_csv(gh.form_path(DATA_DIR, "train" + in_ext), sep=in_seperator, names=INPUT_LABEL_LIST)
    df_bert_train = ensure_bert_data_frame(df_train)

    ## BAD: df_bert_train.to_csv(gh.form_path(DATA_DIR, 'train.tsv'), sep='\t', index=False, header=False)
    ## TODO: df_train.to_csv(gh.form_path(DATA_DIR, 'train.tsv'), sep='\t', index=False, header=False)

    # read source data from csv file
    ## OLD: df_train = pd.read_csv(gh.form_path(DATA_DIR, "train" + in_ext))
    test_columns = [INPUT_LABEL_LIST[0], INPUT_LABEL_LIST[2]]
    df_test = pd.read_csv(gh.form_path(DATA_DIR, "test" + in_ext), sep=in_seperator, names=test_columns)
    df_bert_test = ensure_bert_data_frame(df_test, is_test=True)

    ## TODO: alternative version
    ## #create a new dataframe for train, dev data
    ## df_bert = pd.DataFrame({'guid': df_train['id'],
    ##                         'label': df_train['label'],
    ##                         'alpha': ['a']*df_train.shape[0],
    ##                         'text': df_train['text']})

    #split into test, dev
    # TODO: only do if no det.tsv file
    ## OLD" df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)
    dev_file = gh.form_path(DATA_DIR, "dev" + in_ext)
    if system.file_exists(dev_file):
        df_dev = pd.read_csv(dev_file, sep=in_seperator, names=INPUT_LABEL_LIST)
        df_bert_dev = ensure_bert_data_frame(df_dev)
    else:
        df_bert_train, df_bert_dev = train_test_split(df_bert_train, test_size=0.01)  
        
    ## ALT: create new dataframe for test data
    ## df_bert_test = ensure_bert_data_frame(df_test)
    ## pd.DataFrame({'guid': df_test['id'],
    ##               'text': df_test['text']})

    #output tsv file, no header for train and dev
    if not USE_TSV_INPUT:
        df_bert_train.to_csv(gh.form_path(OUTPUT_DIR, 'train.tsv'), sep='\t', index=False, header=False)
        df_bert_dev.to_csv(gh.form_path(OUTPUT_DIR, 'dev.tsv'), sep='\t', index=False, header=False)
        df_bert_test.to_csv(gh.form_path(OUTPUT_DIR, 'test.tsv'), sep='\t', index=False, header=True)

    ## TODO: work example error from run_classifier.py cusstomization into assertion
    ## label_id = label_map[example.label]
    ## KeyError: '2'`

    ## TODO: Run NVIDIA CUDA utility and make sure capable of running TensorFlow w/ GPU's.
    ## Also, warn is graphics memory is too low.
    ## nvidia-smi
    system.setenv("BERT_BASE_DIR", MODEL_DIR)
    ##CUDA_VISIBLE_DEVICES=0
    ## python script.py
    print("Make sure your GPU Processor has sufficient memory, besides adequate number of units")
    ## BAD: gh.issue("nvidia-smi")
    print(gh.run("nvidia-smi"))
    # note: 0 is the order, not the total number
    system.setenv("CUDA_VISIBLE_DEVICES", "0")
    # TODO: use run and due sanity checks on output; u
    is_lower_case = system.to_string(LOWER_CASE).lower()
    bert_proper_args = ("--vocab_file={md}/vocab.txt".format(md=MODEL_DIR) if (not USE_ALBERT) else "--spm_model_file={md}/albert.model".format(md=MODEL_DIR))
    print(gh.run("{ci} {bpa} --task_name={t} --do_train=true --do_eval=true --do_test=true --data_dir={dd}  --{bn}_config_file={bcf} --init_checkpoint={md}/{bn}_model.ckpt --max_seq_length=64 --train_batch_size=2 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir={od} --do_lower_case={lc} --save_checkpoints_steps 10000", ci=CLASSIFIER_INVOCATION, bpa=bert_proper_args, t=TASK_NAME, dd=DATA_DIR, md=MODEL_DIR, od=OUTPUT_DIR, lc=is_lower_case, bn=BERT_NAME, bcf=BERT_CONFIG_FILE))
    ## sample output:
    ## eval_accuracy = 0.96741855 eval_loss = 0.17597112 global_step = 236962 loss = 0.17553209
    ## model_checkpoint_path: "model.ckpt-236962" all_model_checkpoint_paths: "model.ckpt-198000"all_model_checkpoint_paths: "model.ckpt-208000"all_model_checkpoint_paths: "model.ckpt-218000"all_model_checkpoint_paths: "model.ckpt-228000"all_model_checkpoint_paths: "model.ckpt-236962"
    ## aternative run
    ## CUDA_VISIBLE_DEVICES=0 python run_classifier.py --task_name=cola --do_predict=true --data_dir=./dataset --vocab_file=./model/vocab.txt --bert_config_file=./model/bert_config.json --init_checkpoint=./bert_output/model.ckpt-236962 --max_seq_length=64 --output_dir=./bert_output/
    ## 1.4509245e-05 1.2467547e-05 0.999946361.4016414e-05 0.99992466 1.5453812e-051.1929651e-05 0.99995375 6.324972e-063.1922486e-05 0.9999423 5.038059e-061.9996814e-05 0.99989235 7.255715e-064.146e-05 0.9999349 5.270801e-06
    ## alternative input
    ## # read the original test data for the text and id
    ## df_test = pd.read_csv(gh.form_path(OUTPUT_DIR, 'test.tsv'), sep='\t')
    ## # read the results data for the probabilities
    ## df_result = pd.read_csv('bert_output/test_results.tsv', sep='\t', header=None)
    ## # create a new dataframe
    ## df_map_result = pd.DataFrame({'guid': df_test['guid'],
    ##                               'text': df_test['text'],
    ##                               'label': df_result.idxmax(axis=1)})
    ## # view sample rows of the newly created dataframe
    ## df_map_result.sample(10)
    return

#------------------------------------------------------------------------

if __name__ == '__main__':
    main()
