import time
import RandomForestLearner as RFLearner
import csv as csv
import numpy
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
from scipy.spatial import cKDTree
import cProfile,pstats
import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

split = 600

def TestRandomForest(filename,k=3,draw=0):
    reader= csv.reader(open(filename,'rU'),delimiter=',')
    learner = RFLearner.RandomForestLearner(k)
    i=0
    indata = None
    for row in reader:
        i = i+1
        temp = numpy.zeros([1,3])
        i=0
        for elements in row:
            temp[0][i]=string.atof(elements)
            i=i+1
        if indata is None:
            indata = temp
        else:
            indata = numpy.append(indata,temp,axis=0)
    start = time.clock()
    learner.addEvidence(indata[0:split])
    learner.buildForest()
    traintime = (time.clock() - start)
    print "Train time is ", traintime
    start = time.clock()
    yfitted = numpy.zeros([400])
    for i in range(600,1000):
        yfitted[i-600]=learner.query(indata[i,0:2])
    querytime = (time.clock() - start)/400
    print "Query time is ", querytime
    cormat = numpy.corrcoef(indata[600:1000,2],yfitted)
    print "Correlation coefficient of out sample data is \n",cormat[0][1]
    
    dif = yfitted - indata[600:1000,2]
    RMS = 0
    for err in dif:
        RMS = RMS + err*err
    RMS = numpy.sqrt(RMS/400)
    print "RMS of out sample data is ",RMS

    ytfitted = numpy.zeros([600])
    for i in range(0,600):
        ytfitted[i]=learner.query(indata[i])
    cormatoft = numpy.corrcoef(indata[0:600,2],ytfitted)
    print "Correlation coefficient of in sample data is \n",cormatoft[0][1]
    dif = ytfitted - indata[0:600,2]
    RMSt = 0
    for err in dif:
        RMSt = RMSt + err*err
    RMSt = numpy.sqrt(RMSt/600)
    print "RMS of in sample data is ",RMSt
    if(draw ==1):
        xax = numpy.zeros([400])
        for i in range(600,1000):
            xax[i-600]=i
        plt.plot(xax,yfitted,'ro')
        plt.plot(xax,indata[600:1000,2],'bo')
        plt.show()
            

        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(indata[600:1000,0],indata[600:1000,1],indata[600:1000,2],c='b')
        ax.scatter(indata[600:1000,0],indata[600:1000,1],yfitted[:],c='r')
        plt.show()

    return k,traintime,querytime,cormat[0][1],cormatoft[0][1],RMS,RMSt

def main():
    print "Now Test KNNLearner!"
    knnwriter = csv.writer(file('RFL-classification.csv', 'wb'))
    for k in range(0,100):
        print k
        KNNresult = TestRandomForest('data-classification-prob.csv',k+1)
        knnwriter.writerow(KNNresult)

    knnwriter = csv.writer(file('RFL-ripple.csv', 'wb'))
    for k in range(0,100):
        print k
        KNNresult = TestRandomForest('data-ripple-prob.csv',k+1)
        knnwriter.writerow(KNNresult)



if __name__=="__main__":
    #main() is used to test the KNNLearner and LinRegLearner
    #drawbestk() is used to show the 3D plot and chart of the best K for two data set
    main()

