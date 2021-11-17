#Import all the required packages
import os
import datetime
import click
import matplotlib
import tables

import random as rn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import stats
from math import pi

from quantileNetwork import QuantileNet, sample_net

modelName = "genToReco"
sampleNum=1000
batchSize = 1000




#Set random number seeds
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)    



#Extract data
data = np.load("mlData.npy").T
trainIn = data[8:12,:]
trainIn[0,:] = np.log(trainIn[0,:])
trainIn[3,:] = np.log(trainIn[3,:]+0.7)
trainOut = data[4:8]
trainOut[0,:] = np.log(trainOut[0,:])
trainOut[3,:] = np.log(trainOut[3,:]+0.7)

trainOut=(trainOut+10)/(trainIn+10)

trainIn = trainIn.T
trainOut = trainOut.T

normInfoIn=[[0,1],[0,1],[0,1],[0,1]]
normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
trainIn, testIn, trainOut, testOut = train_test_split(trainIn,
                                                    trainOut,
                                                    test_size=1/3,
                                                    random_state=42)


filename = 'genToReco.h5'
ROW_SIZE = testIn.shape[0]
NUM_COLUMNS = sampleNum*4
f = tables.open_file(filename, mode='w')
atom = tables.Float32Atom()
allData = f.create_earray(f.root, 'data', atom, (0, NUM_COLUMNS))


#Normalization 
for x in [0,1,2,3]:
    normInfoIn[x]=[np.mean(trainIn[:,x]),np.std(trainIn[:,x])]
    normInfoOut[x]=[np.mean(trainOut[:,x]),np.std(trainOut[:,x])]
    testIn[:,x]=(testIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
model = QuantileNet(network_type="not normalizing") #Used to load network
newModel = tf.keras.models.load_model(modelName, model.custom_objects())

for y in range(0, testIn.shape[0], batchSize):
    currentTestIn = testIn[y:min(y+batchSize, testIn.shape[0]),:]
    currentTestIn=tf.transpose(currentTestIn)
    currentTestIn=tf.cast(currentTestIn, tf.float32)
    print(currentTestIn.shape)
    out=sample_net(newModel, #Network
                    sampleNum, #Number of samples
                    currentTestIn, #Input
                    currentTestIn.shape[1], #Number of examples (batch size)
                    currentTestIn.shape[0], #4d input
                    4, network_type="not normalizing") #4d output
    out = np.array(out)
    currentTestIn = np.array(currentTestIn).T
    for x in range(4):
        out[:,:,x] = out[:,:,x]*normInfoOut[x][1]+normInfoOut[x][0]
        currentTestIn[:,x] = currentTestIn[:,x]*normInfoIn[x][1]+normInfoIn[x][0]
    for x in range(sampleNum):
        out[:,x,:] = out[:,x,:]*(10+currentTestIn)-10
    out[:,:,0] = np.exp(out[:,:,0])
    out[:,:,2] = out[:,:,2]%(2*pi)
    out[:,:,2] = np.float32(tf.where(out[:,:,2]>pi, out[:,:,2]-2*pi, out[:,:,2]))
    out[:,:,3] = np.exp(out[:,:,3])-0.7
    ranges=[[0,500],[-5,5],[-5,5],[0,500]]
    out = out.reshape(out.shape[0], out.shape[1]*out.shape[2])
    allData.append(out)
    percentDone=min(y+batchSize, testIn.shape[0])*100/testIn.shape[0]
    percentDone=round(percentDone,4)
    print("Processing is " + str(percentDone)+"% done")
