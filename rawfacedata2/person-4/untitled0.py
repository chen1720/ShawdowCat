# -*- coding: utf-8 -*-
"""
Created on Thu Aug 04 10:51:14 2016

@author: qchen
"""
import os
output_folder = 'C:\Users\qchen\My_Projects\LDR_FacialRecognition\Openface_database\person-4'
fnames = os.listdir(output_folder)

for fname in fnames:
        pnum = fname.split("-")[1].replace(".jpg","")
        os.rename(fname,"image-{}.jpg".format(pnum))
    
    