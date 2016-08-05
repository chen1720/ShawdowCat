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
classifierModel = "/home/qchenldr/Downloads/featuredata/classifier.pkl"
facelist = {'person-1':'Lucas','person-2':'Tapan','person-3':'Qiming','person-4':'Deepali'}

def getRep(bgrImg):
    if bgrImg is None:
        raise Exception("Unable to load frame")
        
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
   
    # Get all faces
    bb = align.getAllFaceBoundingBoxes(rgbImg)
    
    if bb is None:
        return None
    
    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                    96,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
    
    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    
    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))
    
    return reps

def infer(img,classifier):
    with open(classifier, 'r') as f:
        (le, clf) = pickle.load(f)
        
    reps = getRep(img)
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

if __name__=='__main__':
    dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
    networkModel = os.path.join(openfaceModelDir,"nn4.small2.v1.t7")
    imgDim = 96
    
    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(networkModel,
                                 imgDim = 96,
                                 cuda = False)
    
    capture = cv2.VideoCapture(2)
    capture.set(3, 480)
    capture.set(4, 600)
    capture.set(15,-7.0)
    
    confidenceList = []
    
    while(1):
        ret, frame = capture.read()
        persons, confidences = infer(frame, classifierModel)
        print "P: " + str(persons) + "C: " + str(confidences)
        try:
            confidenceList.append('%.2f' % confidences[0])
        except:
            pass
        
        for i,c in enumerate(confidences):
            if c<=0.65:
            	persons[i] = "Unknown"

        if persons is not None or "Unknown":
	    	cv2.putText(frame, "P: {} C: {}".format(facelist[persons[0]],confidences),
	                	(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)
        cv2.imshow('Facerecognition', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


