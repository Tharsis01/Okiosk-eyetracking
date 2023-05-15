import os
import cv2 as cv
import face_recognition
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision
import torch.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class main_y(torch.nn.Module):
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

class sub_x(torch.nn.Module):
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

class main_x(torch.nn.Module):
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

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

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
        self.fc3 = nn.Linear(60, 1)

    def forward(self,x):
        x = self.layer1(x);
        x = self.layer2(x);
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x);
        x = self.fc2(x);
        x = self.fc3(x);

        return x

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

def getEye(model, times = 1,frameShrink = 0.15, coords = (0,0), counterStart = 0, folder = "eyes"):
    os.makedirs(folder, exist_ok=True)
    webcam = cv.VideoCapture(0)
    counter = counterStart
    ims = []

    while counter < counterStart+times:
        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            # right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)

            left_eye = cv.resize(left_eye, dsize=(100, 50))

            # D
            # isplay the image - DEBUGGING ONLY
            cv.imshow('frame', left_eye)
            pred = model(torch.tensor([[left_eye]],dtype=torch.float))
            print(1080*pred.item())

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

def dataLoad(path, want = 0):
    nameList = os.listdir(path)

    totalHolder = []
    dims = [1080,1920]

    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                            torch.tensor([[int((name.split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder


def evaluateModel(model,testSet, sidelen = 1080):
    # model.eval()
    err = 0
    for (im, label) in testSet:
        output = model(im)
        err += abs(output - label.item())
    # model.train()

    return (err/len(testSet)*sidelen)


# X classifiers
mainx = main_x().to(device)
mainx.load_state_dict(torch.load("xModels/174.plt",map_location=device))
mainx.eval()

subx = sub_x().to(device)
subx.load_state_dict(torch.load("xModels/180.plt",map_location=device))
subx.eval()

# Y classifiers
mainy = main_y().to(device)
mainy.load_state_dict(torch.load("yModels/54x1.plt",map_location=device))
mainy.eval()

testy = dataLoad("LRtest",want=1)
testx = dataLoad("LRtest")
print(evaluateModel(mainx, testx))

trainx = dataLoad("LRtrain")
trainy = dataLoad("LRtrain",want=1)

def ensembleX(im):
    modList = [mainx,subx]
    sumn = 0
    for mod in modList:
        sumn += mod(im).item()
    return sumn / len(modList)

