import numpy
import random
import csv as csv
import string
import Queue as qu
class Node:
    def __init__(self,f,s,l,r,y):
        self.Feature = f
        self.SplitValue = s
        self.Left = l
        self.Right = r
        self.Y = y

class RandomForestLearner:
    
    def __init__(self,k=1):
        self.data = None
        self.k = k
        self.forest=None
    def addEvidence(self, dataX, dataY=None):
        if not dataY == None:
            data = numpy.zeros([dataX.shape[0],dataX.shape[1]+1])
            data[:,0:dataX.shape[1]]=dataX
            data[:,(dataX.shape[1])]=dataY
        else:
            data = dataX

        if self.data is None:
            self.data = data
        else:
            self.data = numpy.append(self.data,data,axis=0)
            
    def query(self,testX):
        temp = list()
        for root in self.forest:
            temp.append(self.search(root,testX))
        pre = 0
        i=float(0)
        for v in temp:
            pre = pre+v
            i=i+1
        pre = float(pre/i)
        return pre
    def search(self,root,testX):
        if(root==None):
            print "BUG!!"
            return -2
        while(root.Feature!=-1):
            if(testX[root.Feature]<=root.SplitValue):
                root=root.Left
            else:
                root=root.Right
        return root.Y
        
    def buildTree(self,data):
        if len(data)==1:
            leaf = Node(-1,-1,None,None,data[0][2])
            return leaf
        else:
            f = random.randint(0,1)
            sample = random.sample(data,2)
            s = (sample[0][f]+sample[1][f])*0.5
            leftset = list()
            rightset = list()
            for i in range(0,len(data)):
                if(data[i][f]<=s):
                    leftset.append(data[i])
                else:
                    rightset.append(data[i])            
            root = Node(f,s,None,None,None)
            root.Left = self.buildTree(leftset)
            root.Right = self.buildTree(rightset)
            return root
    def buildForest(self):
        li = self.data.tolist()
        re = list()
        for i in range(0,self.k):
            temp = self.buildTree(li)
            re.append(temp)
        self.forest = re

def test():
    learner = RandomForestLearner(k=10)
    filename = 'data-classification-prob.csv'
    reader= csv.reader(open(filename,'rU'),delimiter=',')
    indata = None
    i=0
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
    learner.addEvidence(indata[:])
    print learner.data
    learner.buildForest()
    print "Let's test now!"
    Xtest = numpy.array([1,1])
    print len(learner.forest)
    for row in indata:
        pre = learner.query(row[0:2])
        print pre-row[2]
    
def printtree(root):
    if root==None:
        return
    q = qu.Queue()
    q.put(root)
    while(q.qsize()!=0):
        temp=q.get()
        if(temp.Feature==-1):
            print temp.Y
        else:
            print temp.Feature, temp.SplitValue
        if(temp.Left!=None):
            q.put(temp.Left)
        if(temp.Right!=None):
            q.put(temp.Right)
def main():
    test()

if __name__=="__main__":
    main()
