import os
import pickle
import cv2

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

modelDir = "/home/qchenldr/openface/models"
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
classifierModel = "/home/qchenldr/Downloads/featuredata2/classifier.pkl"
facelist = {'person-1':'Lucas','person-2':'Tapan','person-3':'Qiming',
	'person-4':'Deepali','person-5':'Yodar','Unknown':'Unknown'}

capture = cv2.VideoCapture(2)
capture.set(3,480)
capture.set(4,600)
align = openface.AlignDlib(dlibFacePredictor)
networkModel = os.path.join(openfaceModelDir,"nn4.small2.v1.t7")
net = openface.TorchNeuralNet(networkModel,
                              imgDim = 96,
                              cuda = False)
confidenceList = []
                              
def infer(reps,classifier):
    with open(classifier, 'r') as f:
        (le, clf) = pickle.load(f)
        
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        persons.append(le.inverse_transform(maxI))
        confidences.append(predictions[maxI])
        if isinstance(clf,GMM):
            dist = np.linalg.nor(rep - clf.means_[maxI])
            print(" + Distance from the mean: {}".format(dist))
            pass
        
    return (persons, confidences)

while(1):
    
    _,frame = capture.read()
    framergb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    bb = align.getAllFaceBoundingBoxes(framergb)
    alignedFaces = []
    x,y,w,h = 0,0,0,0
    for box in bb:
        alignedFaces.append(
            align.align(96,
                        framergb,
                        box,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

        x,y,w,h = box.left(),box.top(),box.width(),box.height()
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if not reps is None:  
        persons, confidences = infer(reps,classifierModel)
        
        print "You are " + str(persons) + " w/ Confidence " + str(confidences)
        
        try:
            confidenceList.append('%.2f' % confidences[0])
        except:
            pass
        
        for i,c in enumerate(confidences):
            c = round(c,2)
            if c<=0.65:
            	persons[i] = "Unknown"
	             
	    cv2.putText(frame, "Name:{};Confidence:{}".format(facelist[persons[0]],c),
		           (x-100, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)

	cv2.imshow('detection',frame)
	    
	if cv2.waitKey(1) & 0xFF ==27:
	    break
    
capture.release()
cv2.destroyAllWindows()
    
