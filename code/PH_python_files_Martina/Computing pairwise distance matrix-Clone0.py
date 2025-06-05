
# coding: utf-8

# In[15]:


import numpy as np
import csv
import dionysus as d
from numpy import genfromtxt
from scipy.spatial.distance import pdist


# In[20]:


Filenames=["pointpattern"+str(i)+".csv" for i in range(20)]
List_diagrams=[]
Distance_matrix=np.zeros((20,20))
print(Distance_matrix)
homology_dimension=1


# In[25]:


for i in range(20):
    raw_data = genfromtxt(Filenames[i], delimiter=',')
    #print(raw_data)
    temp=np.delete(raw_data, 0, 0)
    #print(temp)
    points=np.delete(temp, 0, 1)
    #print(points)
    dists = pdist(points)
   # print(dists)
    f = d.fill_rips(dists, 2, 6)
    #print(f)
    m = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)
    #print(dgms)
    List_diagrams.append(dgms[homology_dimension])


# In[27]:


#need to choose loss function and distance function here
for i in range(len(List_diagrams)):
    for j in range(i):
 #       dist = d.wasserstein_distance(dgms1[1], dgms2[1], q=2)# computes the q-wasserstein distance
        dist= d.bottleneck_distance(List_diagrams[i], List_diagrams[j])# if using the bottleneck distance
     #   print(dist)
        Distance_matrix[i,j]=dist# if sums of distances
#        Distance_matrix=[i,j]=dist*dist #if sums of squared distances
        
        Distance_matrix[j,i]=Distance_matrix[i,j]


# In[28]:


print(Distance_matrix)


# In[29]:


np.savetxt('dist_matrix.csv', Distance_matrix, delimiter=",")

