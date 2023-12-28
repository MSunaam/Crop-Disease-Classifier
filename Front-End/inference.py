import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cpu"
NUM_CLASSES = 22
PATH_TO_MODEL = 'checkpoint_11_epoch_lr=0.000001_2.pth'
CLASSES = ['Cashew_anthracnose', 'Cashew_gumosis', 'Cashew_healthy', 'Cashew_leaf miner', 'Cashew_red rust', 'Cassava_bacterial blight', 'Cassava_brown spot', 'Cassava_green mite', 'Cassava_healthy', 'Cassava_mosaic', 'Maize_fall armyworm', 'Maize_grasshoper', 'Maize_healthy', 'Maize_leaf beetle', 'Maize_leaf blight', 'Maize_leaf spot', 'Maize_streak virus', 'Tomato_healthy', 'Tomato_leaf blight', 'Tomato_leaf curl', 'Tomato_septoria leaf spot', 'Tomato_verticulium wilt']

def loadImage(pathToImage):
    """"
    Takes the path of an image and returns a tensor with batch dimensions.
    """
    image = Image.open(pathToImage)

    transformToTorch = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensorImage = transformToTorch(image).to(DEVICE, dtype=torch.float32)

    batchTensor = tensorImage.unsqueeze(0)

    return batchTensor

def loadModel(pathToModel):
    """
    Takes the path to a .pth file of the (modified) ResNet, loads weights, and returns the model.
    """
    resnet = models.resnet50(pretrained=True)

    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

    resnet = resnet.to(DEVICE)

    checkpoint = torch.load(pathToModel, map_location=torch.device(DEVICE))                                                                              
    resnet.load_state_dict(checkpoint['model_state_dict'])

    return resnet

def inference(pathToImage):
    """"
    Takes the path to an image and returns the predicted class and confidence.
    """
    tensor = loadImage(pathToImage)
    model = loadModel(pathToModel=PATH_TO_MODEL)
    
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        confidenceTensor = torch.nn.Softmax(dim=1)(output).data
        maxProbIdx = torch.argmax(confidenceTensor, dim=1)
        finalPrediction = CLASSES[maxProbIdx]
        print("INSIDE")
    return {'finalPrediction': finalPrediction, 'confidenceLevel': (confidenceTensor.tolist())[0][maxProbIdx]*100}
    
    # return (f"Disease: {finalPrediction} Confidence Level: {(confidenceTensor.tolist())[0][maxProbIdx]*100}%\n")


