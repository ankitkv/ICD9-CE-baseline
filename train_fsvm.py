#!/usr/bin/env python
########################################################
## Code for the following paper:
## 
## Adler Perotte, Rimma Pivovarov, Karthik Natarajan, Nicole Weiskopf, Frank Wood, Noemie Elhadad
## Diagnosis Code Assignment: Models and Evaluation Metrics, JAMIA, to apper
## 
## Columbia University
## Biomedical Informatics
## Author: Adler Perotte
##
########################################################

import argparse
import numpy as np
## This svm light library is packed with this code, but can also be
## obtained here: https://bitbucket.org/wcauchois/pysvmlight
import svmlight
import os


##############################
# Parse command line arguments


parser = argparse.ArgumentParser(description='Train Flat SVM Model.')
parser.add_argument('-training_text', default='data/training_text.data', help='File location containing training text.')
parser.add_argument('-training_codes', default='data/training_codes_na.data', help='File location containing training codes.')
parser.add_argument('-ICD9_tree',  default='data/MIMIC_parentTochild', help='File location containing the ICD9 codes.')
parser.add_argument('-output_directory', default='fsvm_model/',  help='Directory to save svm model files to.')
parser.add_argument('-I', type=int, default=7042, help='Total number of ICD9 codes. Default=7042')
parser.add_argument('-root', type=int, default=5367, help='Index of the root ICD9 code. Default=5367')
parser.add_argument('-D',  type=int, default=20533, help='Number of training documents. Default=20533')

args = parser.parse_args()

##############################
# Load up training data

training_text_filename = args.training_text
training_codes_filename = args.training_codes
parent_to_child_file = args.ICD9_tree
I = args.I
root = args.root
D = args.D

# Load up the tree
PtoC = [[] for i in range(I)]

f = open(parent_to_child_file)
for line in f:
    line = [int(x) for x in line.strip().split('|')]
    PtoC[line[0]].append(line[1])
f.close()
CtoP = [[] for i in range(I)]

f = open(parent_to_child_file)
for line in f:
    line = [int(x) for x in line.strip().split('|')]
    CtoP[line[1]] = line[0]
f.close()


ancestors = [[] for i in range(I)]
for i in range(I):
    #print i
    if i != root:
        anc = CtoP[i]
        while anc != root:
            ancestors[i].append(anc)
            anc= CtoP[anc]
        ancestors[i].append(root)
for i in range(len(ancestors)):
    ancestors[i] = np.array(ancestors[i] + [i])


# Load up the codes
codes = []
inv_codes = [[] for i in range(I)]
with open(training_codes_filename) as f:
    for d, line in enumerate(f):
        line = line.strip().split('|')[1:]
        if len(line)==0:
            line = [root]
        for code in line:
            inv_codes[int(code)].append(d)
        codes.append([int(x) for x in line])

# Load up the text in a sparse list of tuples
text = []
with open(training_text_filename) as f:
    for d,line in enumerate(f):
        line = line.strip().split(',')
        words = [int(x.split(':')[0])+1 for x in line[1:]]
        sort = np.argsort(words)
        counts = [int(x.split(':')[1]) for x in line[1:]]
        text.append(zip(np.array(words)[sort], np.array(counts)[sort]))
text = np.array(text)


def save_classifier(clf, clf_i):
    directory = args.output_directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    svmlight.write_model(clf, os.path.join(directory,str(clf_i)))


#########################################
## Train SVM classifiers

classifiers = {}
for i in range(I):
    if i != root and len(inv_codes[i]) > 0:
        print i, 'of', I
        clf_child = -np.ones(D, np.int)
        clf_child[np.array(inv_codes[i])] = 1
        clf_text = text
        clf_code = clf_child
        training = zip(clf_code, clf_text)
        clf = svmlight.learn(training, type='classification')
        save_classifier(clf, i)

