#Import all the required packages

import matplotlib
import tables

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

filename = 'genToReco.h5'
f = tables.open_file(filename, mode='r')
reco = f.root.recoData[:,:]
gen = f.root.genData[:,:]
medians = f.root.medianPrediction[:,:]
weights = np.where(np.isnan(medians), 0, 1)
medians = np.where(np.isnan(medians), 0, medians)
absoluteErrorMedians = np.abs(reco-medians)
absoluteErrorReco = np.abs(reco-gen)

batchSize=10000
numSamples = 1000
bins=50
totalSamples = f.root.data.shape[0]
ranges=[[170,310],[-5,5],[-3.2,3.2],[0,70]]
ranges= [[0,500],[-5,5],[-3.2,3.2],[0,100]]
allCounts=[]
allEdges=[]
quantCounts=[]
quantEdges=[]
pTMin = 200
pTMax = 205
totalCounts = 0
for x in range(totalSamples//batchSize+1):
    predCounts=[]
    actualBatchSize = min((x+1)*batchSize, totalSamples)- x*batchSize
    recoData = f.root.recoData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    rawRecoData =f.root.rawRecoData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    predData = f.root.data[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    genData = f.root.genData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    quants = f.root.recoQuantile[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    weights = np.ones(genData[:,0].shape)
    totalCounts+=np.sum(weights)
    for z in range(numSamples):
        currentData = predData[:,4*z:4*(z+1)]
        temp = []
        for y in range(4):
            counts, _ = np.histogram(currentData[:,y], range=ranges[y], bins=bins, weights=weights)
            temp.append(counts)
        predCounts.append(temp)
    predCounts = np.array(predCounts)
    for y in range(4):
        recoCounts, edges = np.histogram(recoData[:,y], range=ranges[y], bins=bins, weights=weights)
        genCounts, _ = np.histogram(genData[:,y], range=ranges[y], bins=bins, weights=weights)
        edges = edges[1:]/2+edges[:-1]/2
        qCounts, qEdges = np.histogram(quants[:,y], range=(0,1), bins=bins, weights=weights)
        qEdges = qEdges[1:]
        if(x==0):
            allCounts.append([recoCounts, predCounts[:,y,:], genCounts])
            allEdges.append(edges)
            quantCounts.append([qCounts])
            quantEdges.append(qEdges)
        else:
            allCounts[y][0]+=recoCounts
            allCounts[y][1]+=predCounts[:,y,:]
            allCounts[y][2]+=genCounts
            quantCounts[y]+=qCounts

        

print("Total counts", totalCounts)    
print("Total samples", totalSamples)    


font = {'family' : 'serif',
        'size'   : 10}
matplotlib.rc('font', **font)
matplotlib.rcParams.update({
    "text.usetex": True})
labels=["p$_T$ (GeV)", "eta", "phi", "mass (GeV)"]
titles=["pT", "eta", "phi", "mass"]
print(quantCounts[0].shape)
for x in range(4):
    qSums = [quantCounts[x][0,0]]
    for y in range(1, len(qCounts)):
        qSums.append(qSums[-1]+quantCounts[x][0,y])
    fig, (ax1) = plt.subplots(1, 1, figsize=(3.5,2.5))#, gridspec_kw={'height_ratios': [0.5, 1]})
    ax1.scatter(qEdges, bins*quantCounts[x][0,:]/np.sum(quantCounts[x][0,:]), c="#d7301f", label="predicted")
    ax1.plot(qEdges, qEdges*0+1, c="k", linestyle="--", label="target")
    ax1.set_ylim((0.7,1.3))
    ax1.set_ylabel("p(quantile)")
    ax1.set_xlabel("quantile")
    ax1.set_title(labels[x])
    ax1.legend()
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.1)
    fig.savefig("calibration_g2r_"+titles[x].replace(" ","_")+".pdf")
    fig.show()
    
    
pred50 = []
pred16 = []
pred84 = []
for x in range(4):
    pred50.append(np.quantile(allCounts[x][1], 0.5, axis=0))
    pred16.append(np.quantile(allCounts[x][1], 0.16, axis=0))
    pred84.append(np.quantile(allCounts[x][1], 0.84, axis=0))


for x in range(4):      
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
    
    ax1.step(allEdges[x], allCounts[x][0], where="mid", color="k", linewidth=0.5)
    ax1.step(allEdges[x], pred50[x], where="mid", color="#d7301f", linewidth=0.5)
    
    ax1.scatter(allEdges[x], pred50[x], label="predicted", color="#d7301f", marker="x", s=5, linewidth=0.5)
    ax1.scatter(allEdges[x], allCounts[x][0], label="reco", color="k", marker="+", s=5, linewidth=0.5)
    
    ax1.set_xlim(ranges[x])
    ax1.set_ylim(0, max(allCounts[x][0])*1.1)
    ax1.set_ylabel("counts")
    ax1.set_xticklabels([])
    ax1.legend()
    
    
    ax2.scatter(allEdges[x], allCounts[x][0]/allCounts[x][0], color="k", marker="+", s=5, linewidth=0.5)
    ax2.errorbar(allEdges[x], pred50[x]/allCounts[x][0], xerr=0, yerr=[pred50[x]/allCounts[x][0]- pred16[x]/allCounts[x][0], pred84[x]/allCounts[x][0]-pred50[x]/allCounts[x][0]], color="#d7301f", ls="",
                capsize=2,capthick=0.5, marker="x", linewidth=0.5, markersize=np.sqrt(5))
    
    ax2.set_xlabel(labels[x])
    ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
    ax2.set_ylim((0.9,1.1))
    ax2.set_xlim(ranges[x])
    
    plt.tight_layout()
    fig.savefig("mD_g2r_"+titles[x].replace(" ","_")+".pdf")
    fig.subplots_adjust(wspace=0.0, hspace=0.1)
    fig.show()
