
# coding: utf-8

# In[3]:


import numpy as np
dist_matrix=np.array([[0,4,3,2,5,1],[4,0,4,4,3,2],[3,4,0,5,6,4],[2,4,5,0,2,3],[5,6,3,2,0,1],[1,2,4,3,1,0]])


# In[11]:


#from a csv file of pairwise distances
import numpy as np
from numpy import genfromtxt
filename="dist_matrix.csv"
dist_matrix = genfromtxt(filename, delimiter=',')
#temp=np.delete(raw_data, 0, 0)
#dist_matrix=np.delete(temp, 0, 1)


# In[12]:


#compute_p_value from (dist_matrix, n, m, number_permutations):
#n is the number of type 1 objects
#m is the number of type 2 objects

number_of_permutations=100
n=10
m=10


rank=1.0

OriginalCost1=sum(dist_matrix[i,j] for i in range(n) for j in range(n))
OriginalCost2=sum(dist_matrix[i,j] for i in range(n,m+n) for j in range(n,m+n))
OriginalCost=OriginalCost1+OriginalCost2


Perm = [[n] for n in range(n+m)]
shuffledTotalDistance=[]
    

for k in range(number_of_permutations-1):
    np.random.shuffle(Perm)
    shuffledCost1=sum(dist_matrix[Perm[i],Perm[j]] for i in range(n) for j in range(n))
    shuffledCost2=sum(dist_matrix[Perm[i],Perm[j]] for i in range(n,n+m) for j in range(n,n+m))
    x=shuffledCost1+shuffledCost2


#This is up to the same positive constant as the Original cost so comparing them will give the same inequality as comparing the scaled versions.
    if x<=OriginalCost:
        rank+=1

rank/=number_of_permutations
print(rank)

