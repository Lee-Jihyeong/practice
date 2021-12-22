# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

iphone = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/iBuy.csv")

#########################
iphone = iphone.drop(columns=['SERIAL'])
iphone['HOUSEHOLD INCOME/Mo.']=np.log(iphone['HOUSEHOLD INCOME/Mo.'])
iphone['AGE']=np.log(iphone['AGE'])
#########################

iphone.head(5)

iphone.info()

iphone.isnull().sum()

iphone.shape

iphone.var()

## HOUSEHOLD INCOME/Mo.    2.292398e+07
format(2.292398e+07,'f')

iphone.std()

iphone.describe()

"""corr, graph"""

def distplot_(df): 
  plt.figure(figsize=(14.4, 8.1))
  plotnumber=1

  for column in df:
      if plotnumber<14:
          ax=plt.subplot(4,4,plotnumber)
          sns.distplot(df[column])
          plt.xlabel(column, fontsize=20)
          plt.ylabel('Values', fontsize=20)
          plt.tight_layout()
      plotnumber += 1
  plt.show()

iphone['HOUSEHOLD INCOME/Mo.']=np.log(iphone['HOUSEHOLD INCOME/Mo.'])
iphone['AGE']=np.log(iphone['AGE'])

np.var(iphone[['HOUSEHOLD INCOME/Mo.','AGE']])

plt.figure(figsize=(20, 25))
plotnumber=1

for column in iphone:
    if plotnumber<14:
        ax=plt.subplot(4,4,plotnumber)
        sns.distplot(iphone[column])
        plt.xlabel(column, fontsize=20)
        plt.ylabel('Values', fontsize=20)
    plotnumber += 1
plt.show()

def corr_(df):
  plt.figure(figsize = (16, 8))

  corr = df.corr()
  mask = np.triu(np.ones_like(corr, dtype = bool))
  sns.heatmap(corr, mask = mask, annot = True, fmt = '.2g', linewidths = 1)
  plt.show()

def elbow(df):
    distortions = []
    K = range(1,20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

"""# **클러스터링**"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

x = np.array(iphone["IPHONE 11 PURCHASE"])
y = np.array(iphone["IPHONE 12 PURCHASE"])

iphone_ = iphone.copy()
iphone_ = iphone_.drop(columns=["IPHONE 11 PURCHASE","IPHONE 12 PURCHASE"])

X = np.stack((x,y),axis=1)

elbow(X)

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

y1 = kmeans.labels_

df = pd.DataFrame(columns={'x', 'y', 'cluster'})
df['x'],df['y'],df['cluster'] = x, y, y1

plt.scatter(df['x'],df['y'], c = df['cluster'], alpha=0.5)
plt.show()

pca = PCA(n_components=1)
Z = pca.fit_transform(iphone_)
Z = Z.flatten()
Y1 = np.stack((Z, x), axis=1)
Y2 = np.stack((Z, y), axis=1)

elbow(Y1)

kmeans_pca_11 = KMeans(n_clusters=4).fit(Y1)
y_pca_11 = kmeans_pca_11.labels_

df_pca_11 = pd.DataFrame(columns={'x', 'y', 'cluster'})
df_pca_11['x'],df_pca_11['y'],df_pca_11['cluster'] = Z, x, y_pca_11

plt.scatter(df_pca_11['x'],df_pca_11['y'], c = df_pca_11['cluster'], alpha=0.5)
plt.show()

elbow(Y2)

kmeans_pca_12 = KMeans(n_clusters=5).fit(Y2)
y_pca_12 = kmeans_pca_12.labels_

df_pca_12 = pd.DataFrame(columns={'x', 'y', 'cluster'})
df_pca_12['x'],df_pca_12['y'],df_pca_12['cluster'] = Z, y, y_pca_12

plt.scatter(df_pca_12['x'],df_pca_12['y'], c = df_pca_12['cluster'], alpha=0.5)
plt.show()

pca_2 = PCA(n_components=2)
Z_1= pca_2.fit_transform(iphone)

elbow(Z_1)

kmeans_pca_all = KMeans(n_clusters=3).fit(Z_1)
y_pca_all = kmeans_pca_all.labels_

df_pca_all = pd.DataFrame(data=Z_1,columns={'x', 'y'})
df_pca_all['cluster'] = y_pca_all

plt.scatter(df_pca_all['x'],df_pca_all['y'], c = df_pca_all['cluster'], alpha=0.5)
plt.show()

iphone['cluster'] = y_pca_all

df_1 = iphone[iphone['cluster']==0]
df_2 = iphone[iphone['cluster']==1]
df_3 = iphone[iphone['cluster']==2]

plt.scatter(iphone['IPHONE 11 PURCHASE'],iphone['IPHONE 12 PURCHASE'], c = iphone['cluster'], alpha=0.5)
plt.show()

iphone.describe()

df_1.describe()

df_2.describe()

df_3.describe()

distplot_(df_1)
corr_(df_1)

distplot_(df_2)
corr_(df_2)

distplot_(df_3)
corr_(df_3)

distplot_(iphone)
corr_(iphone)

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

iphone_ = iphone.drop(columns=['cluster'])
pca_h = PCA(n_components=5)
iphone_h= pca_h.fit_transform(iphone_)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
clustering = model.fit(iphone_h)

def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clustering, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

