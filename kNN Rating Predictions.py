#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD

business = pd.read_csv('business.csv', sep=',', usecols=[5, 41, 42, 55, 58, 59], error_bad_lines=False, encoding="utf-8")
business.columns = ['ambiance', 'business id' , 'categories', 'name', 'num reviews','stars']

user = pd.read_csv('users.csv', sep=',', usecols=[0, 17, 18, 20], error_bad_lines=False, encoding="utf-8")
user.columns = ['average rating', 'name', 'review count', 'user id']

review = pd.read_csv('train_reviews.csv', sep=',', usecols=[0, 4, 5, 8], error_bad_lines=False, encoding="utf-8")
review.columns = ['business id', 'review id', 'stars', 'user id']


# In[7]:


business.head()


# In[8]:


user.head()


# In[9]:


review.head()


# In[110]:


#
# CREATE MAPPINGS OF IDs TO OBJECTS
#

userId = {}
for i, row in user.iterrows():
    userId[row[3]] = row
    
businessId = {}
for i, row in business.iterrows():
    businessId[row[1]] = row
    
ratings = {}
for i, rating in review.iterrows():
    ratings[(rating[0], rating[3])] = rating[2]
    
import collections

user2reviews = collections.defaultdict(list)
for i, row in review.iterrows():
    user2reviews[row[3]].append((row[0], row[2]))
    

    


# In[28]:


#
# CREATE A 2D (SPARSE) MATRIX OF BUSINESS BY USER
# MISSING DATA SET TO 0

user_biz_matrix = review.pivot(index = "user id", columns="business id", values="stars").fillna(0)
user_biz_matrix.head()


# In[29]:


#
# NOW CAST THIS SPARSE MATRIX TO A CSR (COMPRESSED SPARSE ROW) MATRIX
user_biz_csr = csr_matrix(user_biz_matrix.values)
user_biz_csr


# In[31]:


#
# LEARN THE MODEL
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm = 'brute')
model_knn.fit(user_biz_csr)


# In[121]:


#
# calculate the distances of the nearest 5 neighbors
query_index = np.random.choice(user_biz_matrix.shape[0])
distances, indices = model_knn.kneighbors(user_biz_matrix[query_index:query_index+1], n_neighbors = 6)


# In[122]:


#
# dislpay the K nearest neighbors
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(user_biz_matrix.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, user_biz_matrix.index[indices.flatten()[i]], distances.flatten()[i]))


# In[118]:


test_data = pd.read_csv('test_queries.csv', sep=',',error_bad_lines=False, encoding="utf-8")
test_data.columns = ['userid', 'biz id']
test_data.head()


# In[ ]:


# find the average of the NN
def neighbor_avg(uid, bid):
    sum = 0
    dists, ins = model_knn.kneighbors(user_biz_matrix[query_index:query_index+1], n_neighbors = 6)
    
    

