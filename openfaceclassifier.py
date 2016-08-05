import time

start = time.time()

import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


modelDir = "/home/qchenldr/openface/models"
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
workDir = "/home/qchenldr/Downloads/featuredata2"


print("Loading embeddings.")
fname = "{}/labels.csv".format(workDir)
labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
labels = map(itemgetter(1),
             map(os.path.split,
                 map(os.path.dirname, labels)))  # Get the directory.
fname = "{}/reps.csv".format(workDir)
embeddings = pd.read_csv(fname, header=None).as_matrix()
le = LabelEncoder().fit(labels)
labelsNum = le.transform(labels)
nClasses = len(le.classes_)
print("Training for {} classes.".format(nClasses))

clf = SVC(C=1, kernel='linear', probability=True)

clf.fit(embeddings, labelsNum)

fName = "{}/classifier.pkl".format(workDir)
print("Saving classifier to '{}'".format(fName))
with open(fName, 'w') as f:
    pickle.dump((le, clf), f)
