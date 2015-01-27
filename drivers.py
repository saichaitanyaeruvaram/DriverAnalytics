import pandas as pd
import sklearn
import os
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist



def returnDifferential(df):
    dfDiff = df - df.shift[-1]
    dfDiff = dfDiff[:-1]
    return dfDiff

def velocitySeq(dfDist):

    velocity = np.array([i*i for i in dfDist['x']])+np.array([i*i for i in dfDist['y']])
    velocity= [math.sqrt(i) for i in velocity]
    return velocity

def normalize(seq):

    return [i/np.median(seq) for i in seq]

df = pd.read_csv('FolderList.csv')
folderList = list(df['folderNames'])
#print folderList

cumul = []

for j in folderList:
    os.chdir('D:/rg/drivers (1)/drivers/'+str(j))

    meanVel = []
    stdVel = []
    maxAccel =[]
    maxDecel = []
    stdevAccel = []

    for i in range(1,201):
        dfCoord = pd.read_csv(str(i)+'.csv')
        dfDist = dfCoord - dfCoord.shift(-1)
        dfDist = dfDist[:-1]

        velocity = velocitySeq(dfDist)
        acceleration = np.diff(np.array(velocity))

        #totalDist.append(sum(velocity))
        maxAccel.append(max([i for i in acceleration if i>0]))
        maxDecel.append(min([i for i in acceleration if i<0]))
        stdevAccel.append(np.std(acceleration))
        meanVel.append(np.mean(velocity))
        stdVel.append(np.std(velocity))
        #break


    """
    meanVel = np.absolute(meanVel - np.median(meanVel))
    meanVel =  [float(i) for i in meanVel/max(meanVel)]

    stdVel = np.absolute(stdVel - np.median(stdVel))
    stdVel =  [float(i) for i in stdVel/max(stdVel)]
    """
    #totalDist = normalize(totalDist)
    meanVel = normalize(meanVel)
    stdVel = normalize(stdVel)
    maxAccel =normalize(maxAccel)
    maxDecel = normalize(maxDecel)
    stdevAccel = normalize(stdevAccel)

    feature_matrix = np.transpose(np.array([meanVel,stdVel,maxAccel,maxDecel,stdevAccel]))

    cluster_center = np.transpose(np.array([np.median(meanVel),np.median(stdVel),np.median(maxAccel),np.median(maxDecel),np.median(stdevAccel)]))

    feature_matrix1 = [[1/i if i <1 else i for i in j] for j in feature_matrix]

    #print feature_matrix1

    distances =  [pdist([feature_matrix1[i],cluster_center]) for i in range(0,len(feature_matrix1))]
    distances=  np.ravel(distances)
    probs = 1- distances/max(distances)
    pd.DataFrame(feature_matrix1).to_csv('Feature_matrix.csv')
    pd.DataFrame(distances).to_csv('Distances.csv')
    pd.DataFrame(probs).to_csv('Probs.csv')
    cumul.extend(probs)
    print len(cumul)/200

df = pd.DataFrame(cumul)

df.to_csv('Final.csv')
#print cosine_similarity([0.1,0.1],[0.1,4.1])