## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
import sys
import operator
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ast

import util

def extract_feats(direc="train", bgs=[], train=True):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    classes = []
    ids = []
    ml = []
    count = 0
    for datafile in os.listdir(direc):
        if count >= 100:
            break
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        layers = gen_thread_sequence(tree)
        for i in xrange(0,len(layers)):
            if len(ml) <= i:
                ml.append([])
            ml[i].append(layers[i])
        count += 1
        # accumulate features
    X = None
    for i in xrange(0, len(ml)):
        if train:
            bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)
            if X is None:
                X = bigram_vectorizer.fit_transform(ml[i])
            else:
                X = sparse.hstack([X,bigram_vectorizer.fit_transform(ml[i])])
            bgs.append(bigram_vectorizer.get_feature_names())
        else:
            bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1,vocabulary=bgs[i])
            print len(bigram_vectorizer.get_feature_names())
            if X is None:
                X = bigram_vectorizer.transform(ml[i])
            else:
                X = sparse.hstack([X,bigram_vectorizer.transform(ml[i])])
    #X,feat_dict = make_design_mat(fds,global_feat_dict)
    print bgs
    for i in bgs:
        print len(i)
    print len(bgs)
    return X, bgs, np.array(classes), ids#feat_dict, np.array(classes), ids

def gen_thread_sequence(tree):
    # Dictionary of threads of form idnumber:[process1, process2,...] 
    thread_seq = {}
    thread_id = None
    in_thread = False
    max_layers = 0
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "thread" and not in_thread:
            in_thread = True
            thread_id = el.attrib['tid']
            if not thread_id in thread_seq:
                thread_seq[thread_id] = {'layer':0, 'seq':[]}
        elif el.tag == "thread" and in_thread:
            in_thread = False
            thread_id = None
        elif in_thread and not thread_id is None:
            seq = thread_seq[thread_id]['seq']
            if not el.tag == 'all_section':
                if el.tag == 'create_thread' or el.tag == 'create_thread_remote':
                    thread_seq[el.attrib['threadid']]={}
                    thread_seq[el.attrib['threadid']]['layer']=thread_seq[thread_id]['layer']+1
                    thread_seq[el.attrib['threadid']]['seq']=seq
                    max_layers = max(max_layers, thread_seq[thread_id]['layer']+1)
                if len(seq) == 0 or not el.tag == seq[-1]:
                    thread_seq[thread_id]['seq'].append(el.tag)
    # Layered is an list of layers, with each layer containing one string of
    # call sequences
    layered = [[]]*(max_layers+1)
    for k,v in thread_seq.iteritems():
        layered[v['layer']].extend(v['seq'])
    for i in xrange(0,len(layered)):
        layered[i] = " ".join(layered[i])
    while len(layered) < 4:
        layered.append("")
    return layered
'''
def gen_thread_sequence(tree):
    # Dictionary of threads of form idnumber:[process1, process2,...] 
    thread_seq = {}
    thread_id = None
    in_thread = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "thread" and not in_thread:
            in_thread = True
            thread_id = el.attrib['tid']
            if not thread_id in thread_seq:
                thread_seq[thread_id] = []
        elif el.tag == "thread" and in_thread:
            in_thread = False
            thread_id = None
        elif in_thread and not thread_id is None:
            if not el.tag == 'all_section' and not el.tag == 'process':
                thread_seq[thread_id].append(el.tag)
    master_list = []
    for k,v in thread_seq.iteritems():
        master_list.extend(v)
    filestring = ' '.join(master_list)
    return filestring
'''

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c

def train(name):

    # extract features
    print "extracting training features..."
    X_train, bgs, Y_train, _ = extract_feats('train',[])
    print "done extracting training features"
    print X_train.shape

    #X_train_train, X_test_train, y_train_train, y_test_train = train_test_split(X_train, Y_train, test_size=0.33,random_state=42)
    # TODO train here, and learn your classification parameters
    print "learning..."
    lr = LogisticRegression(class_weight="balanced",solver="newton-cg",multi_class="multinomial",max_iter=1000)
    lr.fit(X_train, Y_train)

    print "done learning"
    print

    #print lr.score(X_test_train, y_test_train)
    

    print "extracting test features..."
    X_test, bgs, _, test_ids = extract_feats('test', bgs)
    print "done extracting test features"
    print X_test.shape


    # TODO make predictions on text data and write them out
    print "making predictions..."
    preds = lr.predict(X_test)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(preds, test_ids, 'output2.csv')
    print "done!"

## The following function does the feature extraction, learning, and prediction
def main(argv):

    if len(argv) == 1 and argv[0] == 'train':
        train('train')

    elif len(argv) == 2 and argv[0] == 'test':
        test('test', argv[1])

    else:
        print "\"classification_starter.py train <train_directory>\": Train on directory of xml files."
        print "\"classification_starter.py test <test_directory> <output.csv>\": Write predictions on test directory to csv file."
        sys.exit(2)
    

if __name__ == "__main__":
    main(sys.argv[1:])
    
