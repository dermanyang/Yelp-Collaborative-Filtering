#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import sklearn
from sklearn.decomposition import TruncatedSVD
import math

business = pd.read_csv('business.csv', sep=',', usecols=[5, 41, 42, 55, 58, 59], error_bad_lines=False, encoding="utf-8")
business.columns = ['ambiance', 'business id' , 'categories', 'name', 'num reviews','stars']

user = pd.read_csv('users.csv', sep=',', usecols=[0, 17, 18, 20], error_bad_lines=False, encoding="utf-8")
user.columns = ['average rating', 'name', 'review count', 'user id']

review = pd.read_csv('train_reviews.csv', sep=',', usecols=[0, 4, 5, 8], error_bad_lines=False, encoding="utf-8")
review.columns = ['business id', 'review id', 'stars', 'user id']


# In[3]:


business.head()


# In[4]:


user.head()


# In[5]:


review.head()


# In[6]:


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
        


# In[78]:





# In[7]:


import collections

user2reviews = collections.defaultdict(dict)

for i, row in review.iterrows():
    user2reviews[row[3]][row[0]] = row[2]
# user2reviews maps a user to a map of business : rating    
# dict[userId][businessId] = rating


# In[8]:


# just for testing
print(user2reviews['v1zm5ES3dZtn0htQNMr4Gg']['t41_VSBs7akY2POWNtzqxw'])

# print(user2reviews['QGe-bLXLO497G7NfKOFKcA'])


# In[9]:


#
# CREATE A 2D MATRIX OF BUSINESS BY USER USING PIVOT TABLE
# MISSING DATA SET TO 0

user_biz_matrix = review.pivot(index = "user id", columns="business id", values="stars").fillna(0)
user_biz_matrix.head()


# In[10]:


#
# NOW CAST THIS SPARSE MATRIX TO A CSR (COMPRESSED SPARSE ROW) MATRIX
user_biz_csr = csr_matrix(user_biz_matrix.values)


# In[11]:


#
# MAP USERID TO USER_BIZ_MATRIX LOCATION
userid2idx = {}
idx = 0
for userid, row in user_biz_matrix.iterrows():
    userid2idx[userid] = idx
    idx +=1
    #     print('idx:', idx, 'userid', userid)
    #     idx+=1
    #     if idx == 10:
    #         break


# In[12]:


#
# LEARN THE MODEL
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm = 'brute')
model_knn.fit(user_biz_matrix)


# In[13]:


#
# calculate the distances of the nearest 5 neighbors
# query_index = np.random.choice(user_biz_matrix.shape[0])
query_index = 0
distances, indices = model_knn.kneighbors(user_biz_matrix[query_index:query_index+1], n_neighbors = 6)
# print(query_index)


# In[14]:


#
# dislpay the K nearest neighbors
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(user_biz_matrix.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, user_biz_matrix.index[indices.flatten()[i]], distances.flatten()[i]))


# In[17]:


test_data = pd.read_csv('test_queries.csv', sep=',',error_bad_lines=False, encoding="utf-8")
test_data.columns = ['userid', 'bizid']
test_data.head()


# In[ ]:


# initialize return DF
df = {}
df['index'] = []
df['stars'] = []

# iterate through data frame
for i, row in test_data.iterrows():
    # get the index of userid, row['userid']
    if row['userid'] not in userid2idx:
        df['index'].append(i)
        df['stars'].append(businessId[row['bizid']]['stars'])
        continue
        
    index = userid2idx[row['userid']]
    distances, indices = model_knn.kneighbors(user_biz_matrix[index:index+1], n_neighbors = 5) # this takes a long time :|
    total = 0
    total_count = 0
    business = row['bizid']
    for neighbor in range(1, len(distances.flatten())):
        # for each nearest neighbor, add the prediction
        user = user_biz_matrix.index[indices.flatten()[neighbor]]
        dic = user2reviews[user]
        if business in dic: 
            # a similar yelper has visitied this specific restaurant
            total += dic[business]
            total_count +=1
    if total != 0:
        df['index'].append(i)
        df['stars'].append(total/total_count)
    else:
        df['index'].append(i)
        df['stars'].append(businessId[business]['stars'])  
    
    # to track progression 
    percent = str(i / 50079 * 100)[0:4]
    print('iteration #', i, '...', percent, '% complete')


# In[94]:


df = pd.DataFrame(data=df)
df.to_csv('submission.csv')

