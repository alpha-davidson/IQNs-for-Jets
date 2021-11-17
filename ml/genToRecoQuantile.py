""" 
Training code for the raw to gen quantile network. This network is used to
go from a raw reco jet to a gen jet
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #supress tensorflow warning messages
warnings.filterwarnings("ignore", category=FutureWarning) 

import click

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import random as rn

from sklearn.model_selection import train_test_split

from quantileNetwork import QuantileNet, make_dataset

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)

@click.command()
@click.option('--hidden', default=5, help='Number of hidden layers')
@click.option('--width', default=50, help='Width of the hidden layers')
@click.option('--alpha', default=0.2, help='Slope for leaky relu')
@click.option('--initialLR', default=0.001, help='initial learning rate')
@click.option('--batch', default=512, help='batch size')
@click.option('--cycles', default=4, help='Number of cylces to train for')
@click.option('--epochs', default=100, help='Number of epochs in a cylce')
@click.option('--patience', default=100, help='Number of epochs with no improvement before ending')
@click.option('--dataName', default="mlData.npy", help='Name of input data file')
@click.option('--networkName', default="genToReco", help='Name of network')
def main(hidden, width, alpha, initiallr, batch, cycles, epochs, patience, dataname, networkname):
    data = np.load(dataname).T
    rawRecoData = data[0:4,:]
    recoData = data[4:8,:]
    genData = data[8:12]
    rawRecoData[3,:] = rawRecoData[3,:] 
    rawRecoData[0,:] = np.log(rawRecoData[0,:])
    rawRecoData[3,:] = np.log(rawRecoData[3,:]+0.7)
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+0.7)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+0.7)

    inputData=genData
    outputData=(recoData+10)/(inputData+10)

    trainIn = inputData.T
    trainOut = outputData.T

    trainIn, testIn, trainOut, testOut = train_test_split(trainIn,
                                                        trainOut,
                                                        test_size=1/3,
                                                        random_state=42)
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]
    normInfoOut=[[0,1],[0,1],[0,1],[0,1]]

    for x in [0,1,2,3]:
        testIn[:,x]=(testIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        trainIn[:,x]=(trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]
        testOut[:,x]=(testOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))

        trainOut[:,x]=(trainOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
    
    x_val, y_val = make_dataset(trainIn, #input x
          trainOut, #input y
          4, # x dims
          4, # y dims
          trainIn.shape[0]) # examples
    print(x_val.shape, y_val.shape)
    trainIn, valIn, trainOut, valOut = train_test_split(x_val,
                                                        y_val,
                                                        test_size=1/10,
                                                        random_state=42)

    print(trainIn.shape, valIn.shape, trainOut.shape, valOut.shape)
    model = QuantileNet(network_type="not normalizing")

    model.add(
        tf.keras.layers.Dense(
            width,
            kernel_initializer="glorot_uniform",
            activation=None
            ))
    model.add(tf.keras.layers.LeakyReLU())
    

    for n in range(hidden - 1):
        model.add(
            tf.keras.layers.Dense(
                width,
                kernel_initializer="glorot_uniform",
                activation=None))
        model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Dense(
            1,
            kernel_initializer="glorot_uniform",
            activation=None))
   
    callbackMetric="val_loss"
    callback = tf.keras.callbacks.EarlyStopping(
            monitor=callbackMetric, patience=patience, restore_best_weights=True)
    trainOut = tf.expand_dims(trainOut,1)
    valOut = tf.expand_dims(valOut,1)

    epochList=[]
    trainLossList=[]
    valLossList=[]
    for x in range(cycles):
        model.compile(optimizer=tf.keras.optimizers.Adam(initiallr * (10**(-x)),
                      amsgrad=True),
                      loss=model.loss,
                     run_eagerly=False)


        history = model.fit(
            trainIn,
            trainOut,
            validation_data=(
                valIn,
                valOut),
            epochs=epochs,
            batch_size=batch,
            verbose=2,
            callbacks=[callback])#, callback2])
        
        trainLoss = history.history["loss"]
        valLoss = history.history["val_loss"]
        epochList.append(len(trainLoss))
        trainLossList.append(trainLoss)
        valLossList.append(valLoss)
        #Save the network
        model.save(networkname, save_traces=False)
        
        
        
    epochList = range(1,sum(epochList)+1)
    trainLoss = np.concatenate(trainLossList, axis=0)
    valLoss = np.concatenate(valLossList, axis=0)
    plt.figure()
    plt.plot(epochList[1:], trainLoss[1:], label="trainLoss")
    plt.plot(epochList[1:], valLoss[1:], label="valLoss")
    plt.legend()
    plt.savefig(networkname+"_loss_curve.png")
    plt.close()


    plt.figure()
    plt.plot(epochList[1:], np.log(trainLoss[1:]), label="trainLoss")
    plt.plot(epochList[1:], np.log(valLoss[1:]), label="valLoss")
    plt.legend()
    plt.savefig(networkname+"_log_loss_curve.png")
    plt.close()
    print("Evaluating model", networkname, "on the test set.")
    x = [testOut[:,0]]
    y = [testOut[:,1]]
    z = [testOut[:,2]]
    w = [testOut[:,3]]


    inVal = np.zeros(shape=np.array(x).shape)
    dataSetW = np.concatenate([testIn.T, inVal+1, inVal, inVal, inVal, inVal, inVal, inVal, x], axis=0)
    dataSetZ = np.concatenate([testIn.T, inVal, inVal+1, inVal, inVal, x, inVal, inVal, y], axis=0)
    dataSetY = np.concatenate([testIn.T, inVal, inVal, inVal+1, inVal, x, y, inVal, z], axis=0)
    dataSetX = np.concatenate([testIn.T, inVal, inVal, inVal, inVal+1, x, y, z, w], axis=0)
    dataSet = np.concatenate([dataSetW, dataSetZ, dataSetY, dataSetX], axis=1)


    testIn = dataSet[:-1,:].T
    testOut = np.expand_dims(dataSet[-1,:],1)
    model.evaluate(testIn, testOut, verbose=2, batch_size=131072)


if __name__ == '__main__':
    main()