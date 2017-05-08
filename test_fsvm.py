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


parser = argparse.ArgumentParser(description='Test Hierarchical SVM Model.')
parser.add_argument('-testing_text',  default='data/testing_text.data', help='File location containing testing text.')
parser.add_argument('-ICD9_tree', default='data/MIMIC_parentTochild',  help='File location containing the ICD9 codes.')
parser.add_argument('-output_predictions_file', default='fsvm_predictions.data',  help='File to save predictions to.')
parser.add_argument('-classifier_directory', default='fsvm_model/', help='Directory trained SVMs were saved to.')
parser.add_argument('-I', type=int, default=7042, help='Total number of ICD9 codes. Default=7042')
parser.add_argument('-root', type=int, default=5367, help='Index of the root ICD9 code. Default=5367')
parser.add_argument('-D',  type=int, default=20533, help='Number of testing documents. Default=2282')

args = parser.parse_args()


##############################
# Load up training data

testing_text_filename = args.testing_text
parent_to_child_file = args.ICD9_tree
I = args.I
root = args.root
TD = args.D
clf_directory = args.classifier_directory

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


# Create ancestors array
ancestors = [[] for i in range(I)]
children = PtoC[root]
for child in children:
    ancestors[child].append(root)
while len(children) > 0:
    new_children = []
    for child in children:
        for gc in PtoC[child]:
            ancestors[gc].extend([child] + ancestors[child])
            new_children.append(gc)
    children = new_children
for i in range(len(ancestors)):
    ancestors[i] = np.array(ancestors[i] + [i])

# Load up the text in a sparse list of tuples
test_text = []
with open(testing_text_filename) as f:
    for d,line in enumerate(f):
        line = line.strip().split(',')
        words = [int(x.split(':')[0])+1 for x in line[1:]]
        sort = np.argsort(words)
        counts = [int(x.split(':')[1]) for x in line[1:]]
        test_text.append(zip(np.array(words)[sort], np.array(counts)[sort]))

predictions_by_code = {}

def load_classifier(clf_i):
    clf = svmlight.read_model(os.path.join(clf_directory,str(clf_i)))
    return clf


########################################
## Make predictions for each of the ICD9 codes

classifiers = {}
for i in range(I):
    if os.path.exists(os.path.join(clf_directory,str(i))):
        print i
        clf = load_classifier(i)
        preds = np.array(svmlight.classify(clf, zip(np.zeros(len(test_text)),test_text)))
        predictions_by_code[i] = [j for j in range(len(preds)) if preds[j] > 0]

# Invert predictions
predictions = [[root] for x in range(TD)]
for code, docs in predictions_by_code.iteritems():
    for doc in docs:
        predictions[doc].append(code)

# Output predictions
with open(args.output_predictions_file, 'w') as f:
    for prediction_set in predictions:
        f.write('|'.join([str(x) for x in prediction_set]) + '\n')
