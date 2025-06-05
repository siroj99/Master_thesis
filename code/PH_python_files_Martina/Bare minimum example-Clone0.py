
# coding: utf-8

# # The bare minimum to get persistence diagrams from an input filtration

# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
#These are various packages you need to run the code. 
#This will be at the top of any computations you do (just copy and paste). 


# In[31]:


simplices = [([0], -1), ([1],0),([2],3),([3],5),([0,3],5),([2,3],5),([4],7),([3,4],7),([1,4],7)]
#This is the spot you can put whatever in the filtration of simplicial complexes you have
#Everything else below stays the same.
f = d.Filtration()
for vertices, time in simplices:
    f.append(d.Simplex(vertices, time))
    f.sort()


# In[32]:


#Include this if you want to see the have a list in filtration as a list of simplicies in chronological order.
for s in f:
    print(s)


# In[33]:


m = d.homology_persistence(f)
dgms = d.init_diagrams(m, f)
print(dgms)
for i, dgm in enumerate(dgms):
    for pt in dgm:
        print(i, pt.birth, pt.death)


# In[34]:


d.plot.plot_diagram(dgms[0], show = True)
d.plot.plot_bars(dgms[0], show = True)

