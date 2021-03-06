import os
import pickle
import cv2

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
from Tkinter import *

class getname:
   def __init__(self,top):
       self.name = ""
       frame_e = Frame(top)
       frame_e.pack()
       self.t_name = StringVar()
       text = Entry(frame_e,textvariable=self.t_name, bg="white")
       text.pack()
       L1 = Label(frame_e,text='Please enter your name')
       L1.pack(side=TOP)
       nameButton = Button(frame_e, text="Predict!", command=self.Naming)
       nameButton.pack(side=BOTTOM, anchor=S)
   def Naming(self):
       self.name = self.t_name.get()
       root.destroy()
       
root = Tk()
root.geometry=("100x100+500+100")
D=getname(root)
root.mainloop()

modelDir = "/home/qchenldr/myproject/openface/models"
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
classifierModel = "/home/qchenldr/myproject/featuredata2/classifier.pkl"
objs = os.listdir("/home/qchenldr/myproject/aligndata2")

facelist = {'person-1':'Lucas','person-2':'Tapan','person-3':'Qiming',
	'person-4':'Deepali','person-5':'Yodar','person-6':'Sean','Unknown':'Unknown'}

for obj in objs:
    if not obj in facelist and obj.startswith("person"):
        newperson = obj
	facelist[newperson] = D.name

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
    x,y,w,h = [],[],[],[]
    conf = []
    for box in bb:
        alignedFaces.append(
            align.align(96,
                        framergb,
                        box,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if not reps is None:  
        persons, confidences = infer(reps,classifierModel)
        
        print "You are " + str(persons) + " w/ Confidence " + str(confidences)

        for j,c in enumerate(confidences):
            conf.append(round(c,2))
            if c<=0.70:
                persons[j] = "Unknown"

        for i,box in enumerate(bb):
            x.append(box.left())
            y.append(box.top())
            w.append(box.width())
            h.append(box.height())
            cv2.rectangle(frame, (x[i],y[i]),(x[i]+w[i],y[i]+h[i]),(255,128,0),2)
            cv2.putText(frame, "Hello {} {}".format(facelist[persons[i]],conf[i]),
                        (x[i]-100, y[i]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,128,0),2)

	cv2.imshow('detection',frame)
	    
	if cv2.waitKey(1) & 0xFF ==27:
	    break
    
capture.release()
cv2.destroyAllWindows()
    
