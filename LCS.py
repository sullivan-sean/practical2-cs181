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
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import preprocessing
import ast
import bisect
import matplotlib.pyplot as plt
%matplotlib inline
malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap",
           "Lipler", "Magania", "None", "Poison", "Swizzor", "Tdss",
           "VB", "Virut", "Zbot"]

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

def mlcs(strings):
    """Return a long common subsequence of the strings.
    Uses a greedy algorithm, so the result is not necessarily the
    longest common subsequence.

    """
    if not strings:
        raise ValueError("mlcs() argument is an empty sequence")
    strings = list(set(strings)) # deduplicate
    alphabet = set.intersection(*(set(s) for s in strings))

    # indexes[letter][i] is list of indexes of letter in strings[i].
    indexes = {letter:[[] for _ in strings] for letter in alphabet}
    for i, s in enumerate(strings):
        for j, letter in enumerate(s):
            if letter in alphabet:
                indexes[letter][i].append(j)

    # pos[i] is current position of search in strings[i].
    pos = [len(s) for s in strings]

    # Generate candidate positions for next step in search.
    def candidates():
        for letter, letter_indexes in indexes.items():
            distance, candidate = 0, []
            for ind, p in zip(letter_indexes, pos):
                i = bisect.bisect_right(ind, p - 1) - 1
                q = ind[i]
                if i < 0 or q > p - 1:
                    break
                candidate.append(q)
                distance += (p - q)**2
            else:
                yield distance, letter, candidate

    result = []
    while True:
        try:
            # Choose the closest candidate position, if any.
            _, letter, pos = min(candidates())
        except ValueError:
            return ''.join(reversed(result))
        result.append(letter)

def is_subseq(x, y):
    it = iter(y)
    return all(c in it for c in x)

def preprocess_data():
	# preprocess data 
	ml = [[]]*15
	classes = []
	ids = []
	direc = "train/train"
	for datafile in os.listdir(direc):
	    # extract id and true class (if available) from filename
	    id_str,clazz = datafile.split('.')[:2]
	    ids.append(id_str)
	    # add target class if this is training data
	    try:
	        classes.append(malware_classes.index(clazz))
	    except ValueError:
	        # we should only fail to find the label in our list of malware classes
	        # if this is test data, which always has an "X" label
	        assert clazz == "X"
	        classes.append(-1)
	    # parse file as an xml document
	    tree = ET.parse(os.path.join(direc,datafile))
	    ml[malware_classes.index(clazz)] = ml[malware_classes.index(clazz)] + [[gen_thread_sequence(tree)]]

	# make dictionary of system calls and assign character
	sys_calls_dict = {}
	counter = 0
	for clazz in ml:
	    for fil in clazz:
	        for command in fil[0].split(" "):
	            if command not in sys_calls_dict:
	                sys_calls_dict[command] = chr(counter)
	                counter+=1

	ml_sym = [[]]*15
	for clazz_idx,clazz in enumerate(ml):
	    fil_ss = []
	    for fil in clazz:
	        fil_s = ""
	        for command in fil[0].split(" "):
	            fil_s += sys_calls_dict[command]
	        fil_ss += [[fil_s]]
	    ml_sym[clazz_idx] = fil_ss

	# Generate LCS for each class
	claz = []
	for clazz in ml_sym:
	    claz += [[x for x in mlcs([x[0] for x in clazz])]]

	return sys_calls_dict, claz

if __name__ == '__main__':

	sys_calls_dict, claz = preprocess_data()

	classes_new = []
	ids_new = []
	ml_new = [[]]
	direc = 'train/train'
	X_fin = []
	count = 0
	ml = []
	bgs=[]
	for idx,datafile in enumerate(os.listdir(direc)):
	    # extract id and true class (if available) from filename
	    id_str,clazz = datafile.split('.')[:2]
	    ids_new.append(id_str)
	    # add target class if this is training data
	    try:
	        classes_new.append(malware_classes.index(clazz))
	    except ValueError:
	        # we should only fail to find the label in our list of malware classes
	        # if this is test data, which always has an "X" label
	        assert clazz == "X"
	        classes_new.append(-1)
	    # parse file as an xml document
	    tree = ET.parse(os.path.join(direc,datafile))
	    full = gen_thread_sequence(tree)
	    fil_s = []
	    X_data_point = []
	    for command in full.split(" "):
	        fil_s += [sys_calls_dict[command]]
	    for clazz in claz:
	        if is_subseq(clazz, fil_s):
	            X_data_point += [1]
	        else:
	            X_data_point += [0]
	    X_fin += [X_data_point]
	    
	    ml.append(gen_thread_sequence(tree))
	bigram_vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b', min_df=1)
	X = bigram_vectorizer.fit_transform(ml).toarray()
	bgs = bigram_vectorizer.get_feature_names()
	X_fin = np.concatenate((X, X_fin), axis=1)

	# train on X_fin and classes_new


	classes_test = []
	ids_test = []
	ml_test = [[]]
	direc = 'test'
	X_fin_test = []
	count = 0
	ml = []
	for idx,datafile in enumerate(os.listdir(direc)):
	    print idx
	    # extract id and true class (if available) from filename
	    id_str,clazz = datafile.split('.')[:2]
	    ids_test.append(id_str)
	    # add target class if this is training data
	    try:
	        classes_test.append(malware_classes.index(clazz))
	    except ValueError:
	        # we should only fail to find the label in our list of malware classes
	        # if this is test data, which always has an "X" label
	        assert clazz == "X"
	        classes_test.append(-1)
	    # parse file as an xml document
	    tree = ET.parse(os.path.join(direc,datafile))
	    full = gen_thread_sequence(tree)
	    fil_s = []
	    X_data_point = []
	    for command in full.split(" "):
	        if command not in sys_calls_dict:
	            sys_calls_dict[command] = chr(len(sys_calls_dict))
	        fil_s += [sys_calls_dict[command]]
	    for clazz in claz:
	        if is_subseq(clazz, fil_s):
	            X_data_point += [1]
	        else:
	            X_data_point += [0]
	    X_fin_test += [X_data_point]
	    ml.append(gen_thread_sequence(tree))
	    
	bigram_vectorizer = CountVectorizer(ngram_range=(1,1),token_pattern=r'\b\w+\b',min_df=1,vocabulary=bgs)
	X = bigram_vectorizer.transform(ml).toarray()

	X_fin_test = np.concatenate((X, X_fin_test), axis=1)

	# predict on X_fin_test!