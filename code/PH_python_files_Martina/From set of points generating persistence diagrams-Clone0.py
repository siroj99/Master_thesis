
# coding: utf-8

# # Generating the persistent homology of a Rips filtration from a symmetric matrix 

# In[96]:


#From manually putting in a set of points
import numpy as np
points=np.array([[0,0],[2,3],[2,1],[3,4],[4,4],[3,1],[0,2],[4,2]])
x=points[:,0]
y=points[:,1]
print(x)


# In[11]:


#when reading in the points in R^d from a text file, each row one point
#first row ignored
# the point label x1-coord x2-coord ... xd-coord
import numpy as np
import csv

filename="pointpattern.csv"
dimension=2 #usually 2 for examples here
from numpy import genfromtxt
raw_data = genfromtxt(filename, delimiter=',')
temp=np.delete(raw_data, 0, 0)
points=np.delete(temp, 0, 1)
print(points)


# In[97]:


import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.show()


# In[12]:


import dionysus as d
from scipy.spatial.distance import pdist
dists = pdist(points)
#X=np.asanyarray(dists, dtype=float)
print(dists)
f = d.fill_rips(dists, 2, 6)
#the 2 means up to triangles
#the 6 is the max distance for adding edges
print(f)


# In[13]:


m = d.homology_persistence(f)
dgms = d.init_diagrams(m, f)
print(dgms)
for i, dgm in enumerate(dgms):
    for pt in dgm:
        print(i, pt.birth, pt.death)


# In[14]:


d.plot.plot_diagram(dgms[0], show = True)
d.plot.plot_bars(dgms[0], show = True)


# In[15]:


d.plot.plot_diagram(dgms[1], show = True)
d.plot.plot_bars(dgms[1], show = True)

