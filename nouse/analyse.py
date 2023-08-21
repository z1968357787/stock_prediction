import scipy.stats as stats
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import math
import Orange

        
def get_rank(mat):
    ret = []
    #print(type(mat[0][0]))
    for i in range(len(mat)):
        ser = pd.Series (-mat[i])
        ret.append(ser.rank())
    ret = np.array(ret)
    return ret

if __name__ == "__main__":
    data = pd.read_excel("time.xlsx")
    names = ['slave1','slave2','master','Spark']
    data = np.array(data)
    
    data = data[:,1:5]
    data = data.astype('float')
    #print(data.shape)
    print(stats.friedmanchisquare(data[:,0],data[:,1],data[:,2],data[:,3]))

    rank = get_rank(data)
    print(data)
    print(sp.posthoc_nemenyi_friedman(data))
    
    avranks = np.mean(rank,axis=0)
    print(avranks)
    cd = Orange.evaluation.compute_CD(avranks, len(data))
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.savefig("time_CD.png")

