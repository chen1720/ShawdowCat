from os import listdir, rename

import cv2, os, time, random
import numpy as np
from Tkinter import *
import openface
import openface.helper
from openface.data import iterImgs

modelDir = "/home/qchenldr/myproject/openface/models"
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")
outputDir = "/home/qchenldr/myproject/aligndata2"
inputDir = "/home/qchenldr/myproject/rawfacedata2"

i = 0
num = 0
sample = {}
emotion = {}

class getname:
    def __init__(self,top):
        self.name = ""
        frame_e = Frame(top)
        frame_e.pack()
        self.t_name = StringVar()
        text = Entry(frame_e,textvariable=self.t_name, bg="white")
        text.pack()
        L1 = Label(frame_e,text='Please enter your number')
        L1.pack(side=TOP)
        nameButton = Button(frame_e, text="Capture!", command=self.Naming)
        nameButton.pack(side=BOTTOM, anchor=S)
    def Naming(self):
        self.name = self.t_name.get()
        root.destroy()
        
root = Tk()
root.geometry=('+500+300')
D=getname(root)
root.mainloop()
print "Your number is", D.name

align = openface.AlignDlib(dlibFacePredictor)

capture1 = cv2.VideoCapture(2)
rawDir = '/home/qchenldr/myproject/rawfacedata2' 
foldername = "person-{}".format(D.name)
rawDir = os.path.join(rawDir,foldername)
if not os.path.exists(rawDir):
    os.mkdir(rawDir)
print("Initializing...")
time.sleep(2)
print("Please look at the camera..")
time.sleep(2)

while(num<20):
    if num == 5:
        print("Please loot at the right screen")
        time.sleep(2)
    if num == 13:
        print("Please loot at the left screen")
        time.sleep(2)
    
    box = []
    _,frame = capture1.read()
    frameo = frame.copy()
    framergb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    bb = align.getAllFaceBoundingBoxes(framergb)
    x,y,w,h = 0,0,0,0
    for box in bb:
    	x,y,w,h = box.left(),box.top(),box.width(),box.height()
    	cv2.rectangle(frameo, (x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Webcam',frameo)
    filename = "image-{}.png".format(num)
    output_path = os.path.join(rawDir, filename)

    cv2.waitKey(50) & 0xFF

    if box:
        if num > 0:
            cv2.imwrite(output_path, frame)
            num = num + 1
        else:
            num = num + 1
    	continue

capture1.release()
cv2.destroyAllWindows()

os.path.split(output_path)[0]
imgs = list(iterImgs(inputDir))

random.shuffle(imgs)

for imgObject in imgs:
    print("=== {} ===".format(imgObject.path))
    outDir = os.path.join(outputDir, imgObject.cls)
    openface.helper.mkdirP(outDir)
    outputPrefix = os.path.join(outDir, imgObject.name)
    imgName = outputPrefix + ".png"

    if os.path.isfile(imgName):
        print("  + Already found, skipping.")
    else:
        rgb = imgObject.getRGB()
        if rgb is None:
            outRgb = None
        else:
            outRgb = align.align(96, rgb,
                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
                                 skipMulti=None)
            if outRgb is None:
                print("  + Unable to align.")

        if outRgb is not None:
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imgName, outBgr)
