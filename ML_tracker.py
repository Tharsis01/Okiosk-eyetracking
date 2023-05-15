import os
import cv2 as cv
import face_recognition
import numpy as np
import copy
import torch
import torch.nn as nn
import pyautogui
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model loading
class main_y(torch.nn.Module): # main y
    def __init__(self):
        super(main_y, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(f2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 1) 

    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class sub_x(torch.nn.Module): # sub x
    def __init__(self):
        super(sub_x, self).__init__()

        f2 = 8
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(50 * 25 * f2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self,x):
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class main_x(torch.nn.Module): # main x
    def __init__(self):
        super(main_x, self).__init__()

        f1 = 4
        f2 = 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(25 * 12 * f2, 400)
        self.fc2 = nn.Linear(400, 60)
        self.fc3 = nn.Linear(60, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self,x):
        x = self.layer1(x);
        x = self.layer2(x);
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x);
        x = self.fc2(x);
        x = self.fc3(x);
        x = self.fc4(x);

        return x

# class Xmodel(torch.nn.Module):
#     def __init__(self):
#         super(Xmodel, self).__init__()

#         f1 = 4
#         f2 = 16
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=2),

#             nn.ReLU(),
#             nn.BatchNorm2d(f2),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, f1, kernel_size=5, stride=1, padding=2),

#             nn.ReLU(),
#             nn.BatchNorm2d(f1),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc1 = nn.Linear(25 * 12 * f2, 400)
#         self.fc2 = nn.Linear(400, 60)
#         self.fc3 = nn.Linear(60, 1)

#     def forward(self,x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         return x

def maxAndMin(featCoords,mult = 1):
    adj = 10/mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    # print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)


#Preps a color pic of the eye for input into the CNN
def process(im):
    left_eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    left_eye = cv.resize(left_eye, dsize=(100, 50))

    # Display the image - DEBUGGING ONLY
    cv.imshow('frame', left_eye)

    top = max([max(x) for x in left_eye])
    left_eye = (torch.tensor([[left_eye]]).to(dtype=torch.float,
                                              device=device)) / top
    return left_eye

def eyetrack(xshift = 30, yshift=150, frameShrink = 0.15):
    # X classifiers
    mainx = main_x().to(device)
    mainx.load_state_dict(torch.load("my_xModels/178.plt", map_location=device), strict=False)
    mainx.eval()

    subx = sub_x().to(device)
    subx.load_state_dict(torch.load("my_xModels/174.plt", map_location=device), strict=False)
    subx.eval()

    def ensembleX(im): 
        modList = [mainx, subx]
        sumn = 0
        for mod in modList:
            sumn += mod(im).item()
        return sumn / len(modList)

    # Y classifiers
    mainy = main_y().to(device)
    mainy.load_state_dict(torch.load("my_yModels/180.plt",map_location=device))
    mainy.eval()

    webcam = cv.VideoCapture(0)
    
    mvAvgx = []
    mvAvgy = []
    scale = 10
    margin = 200
    margin2 = 50

    while True:
        ret, frame = webcam.read()
        frame = cv.flip(frame,0)
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)
        

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]

            left_eye = process(left_eye)

            x = ensembleX(left_eye)*1080-xshift
            y = mainy(left_eye).item()*1920-yshift

            avx = sum(mvAvgx)/scale
            avy = sum(mvAvgy)/scale
            print(avx,avy)

            mvAvgx.append(x)
            mvAvgy.append(y)

            if len(mvAvgx) >= scale:
                if abs(avx-x) > margin and abs(avy-x)>margin:
                    mvAvgx = mvAvgx[5:]
                    mvAvgy = mvAvgy[5:]
                else:
                    if abs(avx-x) > margin2:
                        mvAvgx = mvAvgx[1:]
                    else:
                        mvAvgx.pop()

                    if abs(avy-y) > margin2:
                        mvAvgy = mvAvgy[1:]
                    else:
                        mvAvgy.pop()
                # else:
                #     mvAvgx = mvAvgx[1:]
                #     mvAvgy = mvAvgy[1:]
                #pyautogui.moveTo(540,960)
                pyautogui.moveTo(avx,avy)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
pyautogui.FAILSAFE=False

eyetrack()