import os
os.environ['DISPLAY'] = ':0';
import numpy as np
import pandas as pd
import cv2 as cv
import os
import argparse
import time
import pickle as pkl
import tensorflow as tf
import keras
from keras.models import model_from_json
import sklearn
import torch 
import torch.nn as nn
from torch.autograd import Variable
import random 
import argparse
import pickle as pkl

import pickle
from torchvision import datasets, transforms, models
from PIL import Image

import matplotlib.pyplot as plt
#from Nose_Detector import Nose_or_Mouth_uncovered
test_transforms = transforms.Compose([transforms.RandomRotation(5),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

net=cv.dnn.readNet("yolov4.weights","./cfg/yolov4.cfg");
classes=[];

with open("./cfg/coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()];


layerNames=net.getLayerNames();
outputLayers=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
#print(outputLayers)

cap=cv.VideoCapture(0);
font=cv.FONT_HERSHEY_PLAIN;

start=time.time();
frameID=0;


frontFaceDetector=cv.CascadeClassifier("./haarcascade_frontalface_default.xml");

logreg_filename = './MODEL/finalized_model.sav'
logreg = pickle.load(open(logreg_filename, 'rb'))
model_ft = models.alexnet(pretrained=True)



while True:
    
    frameID+=1;
    _,img=cap.read();

    grayFrame=cv.cvtColor(img,cv.COLOR_BGR2GRAY);

    height,width,channels=img.shape;
    #(64,64)  (320,320)
    blob=cv.dnn.blobFromImage(img,0.00392,(64,64),(0,0,0),True,crop=False);
    net.setInput(blob);
    outs=net.forward(outputLayers);


    classIdArray=[];
    confidenceArray=[];
    boxesArray=[];
    for out in outs:
        for detection in out:
            scores=detection[5:];
            classID=np.argmax(scores);
            confidence=scores[classID];
            if(confidence>0.1):
                centerX=int(detection[0]*width);
                centerY=int(detection[1]*height);
                w=int(detection[2]*width);
                h=int(detection[3]*height);

                x=int(centerX-w/2);
                y=int(centerY-h/2);
                boxesArray.append([x,y,w,h]);
                confidenceArray.append(float(confidence));
                classIdArray.append(classID);
                #cv.circle(img,(centerX,centerY),10,(0,255,0),2);

    #Do Non maximum supression using IntersectionOverUnion
    #Scores threshold,NMS threshold
    indexes=cv.dnn.NMSBoxes(boxesArray,confidenceArray,0.5,0.4);
    
    for i in range(len(boxesArray)):
        if i in indexes:
            
            x,y,w,h=boxesArray[i];
            label=str(classes[classIdArray[i]]);
            #What classes we detected
            if(label=="person"):
                
                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2);
                cv.putText(img,label,(x,y+30),font,3,(0,0,0),3);

                face=grayFrame[y:y+h,x:x+w];
                faceNoGray=img[y:y+h,x:x+w];

                faceList=frontFaceDetector.detectMultiScale(face);
                print("Person Detected")
                
                for (Nx,Ny,Nw,Nh) in faceList:
                    cv.rectangle(img,(x+Nx,y+Ny),(x+Nx+Nw,y+Ny+Nh),(0,0,0),5);
                    faceNotGray=faceNoGray[Ny:Ny+Nh,Nx:Nx+Nw];
                    

                    face_img = Image.fromarray(faceNotGray)
                    faceTensor = test_transforms(face_img)
                    faceTensor = faceTensor.unsqueeze(0) # batch size 1
                    features = model_ft.features(faceTensor)
                    features = features.view(-1, 6*6*256)
                    feat_df = pd.DataFrame(features.detach().numpy(), columns=[f'img_feature_{n}' for n in range(features.size(-1))])
                    prediction = logreg.predict(feat_df)
                   
                    if(prediction==[0]):
                        cv.putText(img,"Person is wearing a mask.",(x+Nx,y+Ny),font,2,(0,255,0),1);

                    elif(prediction==[1]):
                        cv.putText(img,"Please wear a mask!!!!",(x+Nx,y+Ny),font,2,(0,0,255),1);

                    
                        
                    

    end=time.time();
    elapsed=end-start;
    fps=frameID/elapsed;
    cv.putText(img,"FPS ="+str(round(fps)),(30,30),font,3,(0,0,0),3);
    

    scale_percent = 200 # percent of original size
    widthNew = int(img.shape[1] * scale_percent / 100)
    heightNew = int(img.shape[0] * scale_percent / 100)
    dim = (widthNew, heightNew)
  
    #resize image
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    cv.imshow("Image",img);
    if cv.waitKey(1) & 0xFF == ord('q'):
        break;
    

cap.release();
cv.destroyAllWindows();