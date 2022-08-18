"""
This file is automatically generated by Ipython notebooks
so that pipreqs can automatically read imports
and create requirements.txt
"""


#!/usr/bin/env python
# coding: utf-8

# # Goal of EDA
# 
# The goal of this EDA notebook is to simply get a general look at our data to understand what we're working with. We may find some interesting quirks that may negatively affect our model later on down the line, or help us engineer better features to use. Certain key steps include what each feature represents, getting an idea of what values may be missing, and earning how imbalanced our predicted classes are so our model may be adjusted accordingly.

# In[1]:


import importlib
import JeffUtils
importlib.reload(JeffUtils)


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy
import re
from typing import Union, List


# In[3]:


pd.set_option('display.max_columns', None)


# we set a predetermined random seed so our experiment results can be easily replicated 

# In[4]:


RANDOM_SEED = 1337


# # Loading Raw Data

# Load in data from: https://www.kaggle.com/datasets/abhinav89/telecom-customer?resource=download

# In[5]:


raw_data = pd.read_csv('telecomChurn.zip')


# the data has 100 features to work with, and 100,000 data points

# In[6]:


raw_data.shape


# In[7]:


raw_data.head(5)


# The data description claims that Customer ID is a primary key for this table (no duplicate customer data). We first verify that this is true, and if so, set it as the index

# In[6]:


if raw_data.shape[0] == raw_data['Customer_ID'].nunique():
    raw_data = raw_data.set_index('Customer_ID')
    print('Customer ID is a primary key')


# Cast all column names to lower case and order them alphabetically for ease of use

# In[7]:


raw_data.columns = map(str.lower,raw_data.columns)
raw_data = raw_data[np.sort(raw_data.columns)]


# ## Handling column descriptions

# Since there is literally a hundred columns in this dataset, we may want to group similar columns together and analyze them group by group for a easier time. We just want a quick look so we won't use any complicated state of the art NLP methods, and stick with the tried and true, quick and dirty method of tfidf+kmeans clustering.

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gensim
import spacy


# ### read-in column descriptions

# column descriptions are scraped straight from the kaggle website via beautifulsoup

# In[85]:


col_desc = pd.read_json('columnDescriptions.json',orient='index')[0].rename('col_desc').to_frame()


# In[86]:


#set it to all lower case like we did column names in raw data
col_desc.index = col_desc.index.str.lower()
#pull out customer_id since it is just a primary key
col_desc = col_desc.drop(['customer_id'])
#pull out churn since it is our target variable
col_desc = col_desc.drop(['churn'])


# In[87]:


col_desc.head(5)


# ### Preprocess column descriptions

# In[88]:


#Remove stop words
col_desc['clean_desc'] = (
    col_desc['col_desc']
    .apply(gensim.parsing.preprocessing.remove_stopwords)
)

#Remove characters that are not decimals,
#not letters, and not white space
col_desc['clean_desc'] = (
    col_desc['clean_desc']
    .str.replace('[^a-zA-Z0-9_ \%]','')
)
#Replace double whitespace with single space
#created from operation above
col_desc['clean_desc'] = (
    col_desc['clean_desc']
    .str.replace('  ',' ')
)


# In[89]:


nlp = spacy.load('en_core_web_sm')


# In[90]:


#Tokenize words in the cleaned description and lemmatize them
col_desc['tokenized'] = col_desc['clean_desc'].apply(lambda x: [token.lemma_ for token in nlp(x)])
#Join lemmatized tokens back into single string for ease of use
col_desc['lemmatized'] = col_desc['tokenized'].apply(lambda x: ' '.join(x))


# In[17]:


col_desc.head(5)


# ### Tfidf

# Transform our column descriptions into a tf-idf matrix so that we have numerical data to work with

# In[91]:


tfidf = TfidfVectorizer()
tfidfMatrix = tfidf.fit_transform(col_desc['lemmatized'])


# ### kmeans grouping

# We'll try to cluster similar columns together via K-Means since its quic and dirty.

# In[92]:


#import utility functions
from JeffUtils import graph_cluster_wc, graph_elbow


# We try values from 5-15 clusters for the number of clusters parameter of K-Means. Any less than 5 would likely not yield much separation between each cluster. Meanwhile, any more clusters than 15 is a little much to go through by hand, hence would make this entire process kind of redundant. 

# In[93]:


kmeans = []
for i in range(5,16):
    km = KMeans(n_clusters=i,random_state=RANDOM_SEED)
    km.fit_predict(tfidfMatrix.toarray())
    kmeans.append((i,km.inertia_))


# Then determine the elbow of the loss. The elbow is calculated via the second derivative of the loss curve at every point, and the point with the highest second derivative is chosen as our parameter for the number of clusters. From the graph below, we see that the optimal number of clusters would be 6 clusters.

# In[95]:


loss = pd.DataFrame(kmeans)[1].rename('loss')
#since the min number of clusters tested started at 5
loss.index = loss.index+5
#graph loss at every point and return optimal point
elbow = graph_elbow(loss) 


# we'll use the determined elbow as our n_cluster parameter to run kmeans

# In[96]:


kmeans = KMeans(n_clusters=elbow,random_state=RANDOM_SEED)
col_desc['kmeans_class'] = kmeans.fit_predict(tfidfMatrix.toarray())


# In[97]:


from JeffUtils import graph_all_cluster_wc


# We now graph the 5 most common words appearing in each cluster, with the x axis representing the percentage of column descriptions in the cluster that contains the word in question. Some clusters have clearly defined themes, while others may not. The below summarizes what each column description cluster is likely primarily representing. 
# 
# - cluster 0 doesn't seem to have a unified theme, since the percentage of descriptions sharing each common word is very low. The cluster is therefore likely a collection of descriptions that do not fit well into other clusters. We should check this cluster out later on. 
# - cluster 1 seems to deal with general call statistics
# - cluster 2 deals with statistics relate to minutes
# - cluster 3 deals with monthly statistics
# - cluster 4 deals with statstics of the customer household
# - cluster 5 deals with statistics across the customer's life 

# In[98]:


graph_all_cluster_wc(col_desc,'kmeans_class',figsize=(10,6))


# We can also use T-SNE with default parameters to reduce our high dimensional dataset to two dimensions, so that we may graph our clusters for a quick sanity check. If the graph shows relatively clear demarcation of clusters, it means our "most common word" analysis from before actually represent each cluster well. If certain clusters do not have clear demarcations, we may need to double check that the clusters should actually be separated, and our K-Means didn't make a mistake. 

# In[21]:


from sklearn.manifold import TSNE


# In[22]:


tsne = TSNE(n_components=2,
            perplexity = 30,
            learning_rate='auto',
            init='pca',
            random_state=RANDOM_SEED)
tsneX = tsne.fit_transform(tfidfMatrix.toarray())


# It seems like clusters 1 and 2 have very similar descriptions as their datapoints are pretty intermingled, while the other clusters all have relatively more clearly defined clusters. We should double check that cluster 1 and 2 actually contain descriptions of different topics, and our K-Means didn't separate the descriptions within these two clusters due to some mathematical quirk

# In[27]:


sns.scatterplot(
    x=tsneX[:,0],y=tsneX[:,1],hue=col_desc['kmeans_class'],
    palette=sns.color_palette("tab10")[:col_desc['kmeans_class'].nunique()]
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# create a list of clustered column names for easier access and examination

# In[99]:


def get_clusters_list()->list:
    """returns a list where index corresponds
    to cluster id from kmeans clustering
    """
    clusters = []
    for cluster_id in np.sort(col_desc['kmeans_class'].unique()):
        clusters.append(
            col_desc[col_desc['kmeans_class']==cluster_id].index
        )
    return clusters

def get_col_desc(col_name:Union[str,List[str]])->Union[str,List[str]]:
    """get column description from column name
    """
    return col_desc.loc[col_name]['col_desc']


# In[30]:


clusters = get_clusters_list()


# We randomly sample 5 descriptions from each cluster to avoid being overwhelmed. From these samples, we see:
# - Cluster 1 does indeed deal with general call statistics, while cluster 2 deals with statistics related to minutes.
# - The reason the two clusters are so similar is because they both deal with a variety of similar statistics, but one breaks down said statistics by calls while the other by minutes. Hence, the K-Means was right to cluster them this way.

# In[22]:


get_col_desc(clusters[1]).sample(n=5,random_state=RANDOM_SEED)


# In[23]:


get_col_desc(clusters[2]).sample(n=5,random_state=RANDOM_SEED)


# Lastly, a quick look at cluster 0 does confirm our early suspicion that it is a collection of disparate column descriptions

# In[24]:


get_col_desc(clusters[0]).sample(n=5,random_state=RANDOM_SEED)


# # EDA

# In[31]:


from JeffUtils import jeff_histplot


# ## Missing values

# First we take a look at what values may be missing from our data

# In[32]:


import networkx as nx


# In[33]:


percentage_na = (raw_data.isna().sum().sort_values(ascending=False)/raw_data.shape[0])


# In[34]:


ax = jeff_histplot(percentage_na[percentage_na>0],kde=True)
ax.set_ylim(0,5)


# ## Finding columns with whose missingness is perfectly correlated
# 
# Lets take a look at columns that have a perfectly correlated missing values (i.e if one feature in the group has missing value, the rest would also for that row). 

# In[35]:


#get the correlation of missing values between every column
mia_corr = raw_data.isna().corr()


# In[36]:


perf_corr_cols = (
    mia_corr
    #Since (col1,col2) shows the same relationship as (col2,col1), 
    #and the diagonal will always be perfectly correlated,
    #we just need the upper triangular of the missing matrix
    .where(np.triu(np.ones(mia_corr.shape),1).astype(bool))
    #get columns whose missingness is perfectly correlated 
    .where(mia_corr==1,np.nan)
    #turn into list of tuples
    .stack().index
)


# In[37]:


#create an undirected graph from the data with each column as a node
G = nx.from_pandas_edgelist(perf_corr_cols.to_frame().reset_index(drop=True),0,1)
perf_miss_cols = []

#add each component to a list for easier access
for c in nx.connected_components(G):
    perf_miss_cols.append(c)


# In the cell below we see the groups of columns who have perfectly correlated missingness. This means that for each row in our data, if that row has a column with a missing value, all other columns from the same group would also have a missing value in that row as well. The reason for most columns having perfectly correlated missingness is obvious. For example, the first group all deal with average statistics in the last 6 months, hence if no data is available because they're new customers or their plan was suspended, then they would all have missing values at the same time. 
# 
# One perculiar observation, however, is that the second group deals with comparison with the last 3 months. The immediate reasoning would be that there was a lack of data in the last 3 months. However, the 'avg3mou','avg3qty' columns don't appear perfectly correlated with these missing columns even though they also deal with data in the last 3 months. 
# 

# In[67]:


for c in perf_miss_cols:
    print(c)


# ## Correlations

# Lets take a look at absolute correlation values to see if any particular column already has a high predictive value on churn by itself

# In[482]:


corrs = raw_data.corr()
churn_corrs = corrs['churn']


# Below we see that most of the columns having low correlation values, hence it doesn't seem like any particular column has high predictive power by itself

# In[483]:


#remove first element since it'll be churn's correlation with itself 
sorted_abs_corr = abs(churn_corrs).sort_values(ascending=False)[1:]
jeff_histplot(sorted_abs_corr) 


# It looks like the eqpdays and hnd_price columns have the highest correlation with churn. We'll take a look at these individually

# In[471]:


sorted_abs_corr.head(5)


# Seems like the longer you've been using the same equiment, the more likely you are going to churn

# In[476]:


get_col_desc('eqpdays'),churn_corrs['eqpdays']


# Seems like the more expensive the phone your plan is tied to, the less likely you are to churn

# In[478]:


get_col_desc('hnd_price'),churn_corrs['hnd_price']


# # Categorical Column Examination