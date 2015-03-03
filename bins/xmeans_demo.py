'''
Created on Mar 15, 2012

@author: Wang
'''
import numpy as np
from scipy.cluster.vq import *
import pylab
import matplotlib.pyplot as plt
import cluster


plt.figure()
class1 = np.array(np.random.standard_normal((2,2))) + np.array([5,5]) 
class2 = np.array(np.random.standard_normal((1,2))) 
class3 = np.array(np.random.standard_normal((1,2))) + np.array([-5,-5])
class4 = np.array(np.random.standard_normal((1,2))) + np.array([-5,5])

features = np.vstack((class1,class2,class3,class4))

test = cluster.X_means(features)
print 'From x-means: ', test.final_k
plt.subplot(131)
pylab.plot([p[0] for p in class1],[p[1] for p in class1],'o', markersize = 60)
pylab.plot([p[0] for p in class2],[p[1] for p in class2],'or', markersize = 60)  
pylab.plot([p[0] for p in class3],[p[1] for p in class3],'og', markersize = 60) 
pylab.plot([p[0] for p in class4],[p[1] for p in class4],'ok', markersize = 60) 

class1 = np.array(np.random.standard_normal((2,2))) + np.array([5,5]) 
class2 = np.array(np.random.standard_normal((3,2))) 
#class3 = np.array(np.random.standard_normal((1,2))) + np.array([-5,-5])
#class4 = np.array(np.random.standard_normal((1,2))) + np.array([-5,5])

features = np.vstack((class1,class2))

test = cluster.X_means(features)
print 'From x-means: ', test.final_k
plt.subplot(132)
pylab.plot([p[0] for p in class1],[p[1] for p in class1],'o', markersize = 60)
pylab.plot([p[0] for p in class2],[p[1] for p in class2],'or', markersize = 60)  

class1 = np.array(np.random.standard_normal((1,2))) + np.array([5,5]) 
class2 = np.array(np.random.standard_normal((1,2))) 
class3 = np.array(np.random.standard_normal((1,2))) + np.array([-5,-5])
class4 = np.array(np.random.standard_normal((1,2))) + np.array([-5,5])
class5 = np.array(np.random.standard_normal((1,2))) + np.array([5,-5])

features = np.vstack((class1,class2,class3,class4,class5))

test = cluster.X_means(features)
print 'From x-means: ', test.final_k
plt.subplot(133)
pylab.plot([p[0] for p in class1],[p[1] for p in class1],'o', markersize = 60)
pylab.plot([p[0] for p in class2],[p[1] for p in class2],'or', markersize = 60)  
pylab.plot([p[0] for p in class3],[p[1] for p in class3],'og', markersize = 60) 
pylab.plot([p[0] for p in class4],[p[1] for p in class4],'ok', markersize = 60)
pylab.plot([p[0] for p in class5],[p[1] for p in class5],'om', markersize = 60)

pylab.show()
#