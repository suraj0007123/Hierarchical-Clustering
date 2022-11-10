### Importing the data 
import pandas as pd
#### Reading the data into python
crimedata=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\hierarchical clustering\Dataset_Assignment Clustering\crime_data.csv")

crimedata.shape ## To know the shape of the data 

#EDA analysis
crimedata.describe()

###EDA 
##### Measure of Central Tendency / First of Moment of Business Decision
crimedata.mean()
crimedata.median()
crimedata.mode()

from scipy import stats
stats.mode(crimedata)

# Measures of Dispersion / Second moment business decision
crimedata.var()
crimedata.std()

range=crimedata.iloc[:,1:].max()-crimedata.iloc[:,1:].min()
range

# Third moment business decision
crimedata.skew()
crimedata.kurt()

# Data Visualization
import numpy as np

crimedata.shape

import matplotlib.pyplot as plt
plt.bar(height=crimedata.Murder,x=np.arange(1,51,1))

plt.hist(crimedata.Murder)

plt.bar(height=crimedata["Assault"],x=np.arange(1,51,1))

############## Outlier Treatment ###############
import seaborn as sns
import numpy as np

crimedata.columns

sns.boxplot(crimedata.Murder)

sns.boxplot(crimedata.Assault)

sns.boxplot(crimedata.UrbanPop)

sns.boxplot(crimedata.Rape)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['Rape'])

df_t=winsor.fit_transform(crimedata[['Rape']])
sns.boxplot(df_t.Rape)

duplicate=crimedata.duplicated()
duplicate
sum(duplicate)
crimedata.isna().sum()
crimedata.isnull().sum()

import scipy.stats as stats
import pylab

crimedata.columns

# Checking Whether data is normally distributed
x=crimedata.Assault

stats.probplot(x, dist="norm",plot=pylab)
# Transformation to make workex variable normal
stats.probplot(np.log(x), dist="norm", plot=pylab)

x=crimedata.Murder
stats.probplot(x, dist="norm", plot=pylab)
stats.probplot(np.log(x), dist="norm" , plot=pylab)

x=crimedata.UrbanPop
stats.probplot(x, dist="norm", plot=pylab)
stats.probplot(np.log(x), dist="norm", plot=pylab)

x=crimedata.Rape
stats.probplot(x, dist="norm", plot=pylab)
stats.probplot(np.log(x), dist="norm", plot=pylab)

### Normalization ####
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(crimedata.iloc[:,1:])

df1=df_norm.describe()

### Hierarchical clustering process 
## For Creating Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm, method="complete", metric="euclidean")

## Dendrogram
plt.figure(figsize=(50,50));plt.title("hierarchical clustering dendrogram");plt.xlabel("Index");plt.ylabel("distance")
sch.dendrogram(z,
               leaf_font_size=6,  # rotates the x axis labels
               leaf_rotation=0,   # font size for the x axis labels
)
plt.show()

#### now applying agglomerative clustering, let us choose 11 as clusters from above dendrogram 
from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=11, linkage="complete", affinity="euclidean").fit(df_norm)

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)

crimedata["clust"]=cluster_labels # creating a new column and assigning it to new column 

crimedata

crimedata.shape

df2=crimedata.iloc[:,[5,0,1,2,3,4]]
# Aggregate mean of each cluster
df2.iloc[:,1:].groupby(df2.clust).mean()

df2.head()
