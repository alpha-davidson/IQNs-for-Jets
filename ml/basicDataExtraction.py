import tables
import numpy as np

from sklearn.model_selection import train_test_split

filename = 'rawToGen.h5'
f = tables.open_file(filename, mode='r+')
data = np.load("mlData.npy")
_, data, _, _ = train_test_split(data, data,test_size=1/3, random_state=42)
data = data.T
rawRecoData = data[0:4,:].T
recoData = data[4:8,:].T
genData = data[8:12,:].T
partonFlavors = data[-3:-2,:].T
hadronFlavors = data[-2:-1,:].T
physicsFlators = data[-1:,:].T

f32Atom = tables.Float32Atom()
i32Atom = tables.Int32Atom()

medianPredictionH5 = f.create_earray(f.root, 'medianPrediction', f32Atom,
                                     (0, 4))
meanPredictionH5 = f.create_earray(f.root, 'meanPrediction', f32Atom, (0, 4))
genQuanilteH5 = f.create_earray(f.root, 'genQuantile', f32Atom, (0, 4))
partonFlavorDataH5 = f.create_earray(f.root, 'partonFlavor', i32Atom, (0, 1))
hadronFlavorDataH5 = f.create_earray(f.root, 'hadronFlavor', i32Atom, (0, 1))
physicsFlavorDataH5 = f.create_earray(f.root, 'physicsFlavor', i32Atom,
                                      (0, 1))
genDataH5 = f.create_earray(f.root, 'genData', f32Atom, (0, 4))
recoDataH5 = f.create_earray(f.root, 'recoData', f32Atom, (0, 4))
rawRecoDataH5 = f.create_earray(f.root, 'rawRecoData', f32Atom, (0, 4))


partonFlavorDataH5.append(partonFlavors)
hadronFlavorDataH5.append(hadronFlavors)
physicsFlavorDataH5.append(physicsFlators)
genDataH5.append(genData)
recoDataH5.append(recoData)
rawRecoDataH5.append(rawRecoData)

def calc_quant(data, vals):
        n = data.shape[1]
        left = np.count_nonzero(vals < data, axis=1)
        right = np.count_nonzero(vals <= data, axis=1)
        pct = (right + left + np.where(right>left, 1+0*right, 0*right)) /(2*n)
        return pct
batchSize=10000
numSamples = 1000
bins=100

totalSamples = f.root.data.shape[0]
for x in range(totalSamples//batchSize+1):
    actualBatchSize = min((x+1)*batchSize, totalSamples)- x*batchSize
    data = f.root.data[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    data = data.reshape(data.shape[0]*data.shape[1]//4, 4)
    trueData = f.root.genData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
    
    flavors = partonFlavors[x*batchSize:(x+1)*batchSize]
    
    data = data.reshape((data.shape[0]//actualBatchSize, actualBatchSize, 4),
                        order="f")
    allQuants = []
    allMeans = []
    allMedians = []
    for y in range(4):
        trueVal = data[:,:,y].T
        predVal = trueData[:,y:y+1]
        quants = calc_quant(trueVal, predVal)
        medians = np.median(data[:,:,y], axis=0)
        means = np.mean(data[:,:,y], axis=0)
        allQuants.append(np.expand_dims(quants, axis=1))
        allMedians.append(np.expand_dims(medians, axis=1))
        allMeans.append(np.expand_dims(means, axis=1))

    allQuants = np.concatenate(allQuants, axis=1)
    allMedians = np.concatenate(allMedians, axis=1)
    allMeans = np.concatenate(allMeans, axis=1)
    genQuanilteH5.append(allQuants)
    
    medianPredictionH5.append(allMedians)
    meanPredictionH5.append(allMeans)
    print(str(100*min((x+1)*batchSize, totalSamples)/totalSamples)+"% done")

f.close()
