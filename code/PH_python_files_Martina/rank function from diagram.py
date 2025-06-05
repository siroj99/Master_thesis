
# coding: utf-8

# In[47]:


import numpy as np
import csv
import math
import dionysus as d


# In[16]:


filename="data/2Drand_pts.txt"
dimension=2 #this file has 75 pts in the unit square
points = np.genfromtxt(filename)
print(points.shape)


# In[8]:


import matplotlib.pyplot as plt
x = points[:,0]
y = points[:,1]
plt.scatter(x, y)
plt.show()


# In[11]:


from scipy.spatial.distance import pdist
dists = pdist(points)
#print(dists)
f = d.fill_rips(dists, 2, 1)
#the 2 means up to triangles
#the 6 is the max distance for adding edges
print(f)


# In[12]:


m = d.homology_persistence(f)
dgms = d.init_diagrams(m, f)
print(dgms)


# In[41]:


hdim = 1
for pt in dgms[hdim]:
    print(hdim, pt.birth, pt.death)


# In[18]:


d.plot.plot_diagram(dgms[1], show = True)


# In[27]:


# The following code plots a rank function when the number of PD points isn't too many
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots()
patches = []
# for each (b,d) point, plot a triangle
for pt in dgms[1]:
	b = pt.birth
	d = pt.death
	xy = np.array([[b,b], [b,d], [d,d]])
	triangle = mpatches.Polygon(xy)
	patches.append(triangle)

p = PatchCollection(patches,facecolor='blue',alpha=0.2)
ax.add_collection(p)
ax.axis([0,0.3,0,0.3])
plt.show()


# In[45]:


# the following code evaluates the rank function at a predefined set of values
grid = 0.01 
bvals = np.arange(0.0,0.3,grid) # change the ranges to suit your data
dvals = np.arange(0.0,0.3,grid) 
countKeys = [(b,d) for b in bvals for d in dvals if d >= b ]
rankfun = { ck: 0 for ck in countKeys }
for pt in dgms[1]:
    for (b,d) in countKeys:
        if (b >= pt.birth and d <= pt.death):
            rankfun[(b,d)] += 1
print(len(rankfun))


# In[37]:


def plot_rank(rankFunc,**kwargs):
    # rankFunc is a dictionary of {(birth,death) : number } form
    
    births = set([b for (b,d) in rankFunc.keys()])
    deaths = set([d for (b,d) in rankFunc.keys()])
    
    bvals = list(births)
    dvals = list(deaths)
    bvals.sort()
    dvals.sort()
    
    x_bin_length = bvals[1] - bvals[0]  # this recovers the grid size 
    x_offset = x_bin_length/2.0
    y_bin_length = dvals[1] - dvals[0]
    y_offset = y_bin_length/2.0
    
    x = np.arange(bvals[0]-x_offset,bvals[-1]+x_bin_length,x_bin_length)
    y = np.arange(dvals[0]-y_offset,dvals[-1]+y_bin_length,y_bin_length)
    
    c = np.zeros((len(bvals),len(dvals)))
    for i in range(len(bvals)):
        for j in range(len(dvals)):
            if dvals[j] >= bvals[i]:
                c[i,j] = rankFunc[(bvals[i],dvals[j])]
    
    fig = plt.figure()
    ax  = plt.subplot(111)
    pc = ax.pcolormesh(x,y,c.T,**kwargs)
    ax.set_xlim(x[0],x[-1])
    ax.set_ylim(y[0],y[-1])
    plt.colorbar(pc)
    plt.draw()
    return pc


# In[43]:


pr = plot_rank(rankfun,cmap='Blues',vmin=0, vmax=5)
plt.show()


# In[51]:


# Now calculate a weight function to use in the L2 metric.
# use the same grid spacing, bvals and dvals as for the rankfun earlier. 
# countKeys is the set of (b,d) values with b<=d.  
# you can also choose a parameter A for the exponential decay. 

box_area = grid*grid
A = 1 
weightfun ={(b,d): box_area*math.exp(A*(b-d)) for (b,d) in countKeys }

# The next formula adjusts the weightfn values at the top of the bval, dval domain, 
# to account for essential cycles with d = infinity. 
# It is the sum of weights of all the boxes that would lie above and including (b, max_death)
# We are assuming here that the value of the rank function is constant in this line of boxes, 
# i.e. that there are no PD points with max_death < d < infinity.  
# We are also assuming that all PD points have b >= min(bvals).  
# The formula uses the geometric series
#   sum(box_area*exp(b-max_death-kl))= box_area*exp(b-max_death)*sum(exp(-l)^k)
# over k from 0 to infty. 

max_death = max(dvals)
weightfun.update({(b,max_death): box_area*math.exp(b-max_death)/(1-math.exp(-grid)) for b in bvals})


# In[54]:


L2norm = sum( math.pow(rankfun[key],2)*weightfun[key] for key in rankfun )
print(L2norm)


# In[58]:


# the following computes the weighted L2 distance between two rank functions
# rankA, rankB and weightfun must all be dictionaries with the same keys (as in the above example)
def L2dist(rankA, rankB, weightfun):
    diff = {key: rankA[key]-rankB[key] for key in rankA}
    return sum( math.pow(diff[key],2)*weightfun[key] for key in rankA)

nullrank = {key: 0.0 for key in rankfun}
testL2dist = L2dist(rankfun,nullrank,weightfun)
print(testL2dist) 
    


# In[59]:


def innerproduct(rankA,rankB,weightfun):
    return sum( rankA[key]*rankB[key]*weightfun[key] for key in rankA)

testinnerprod = innerproduct(rankfun,rankfun,weightfun)
print(testinnerprod)

