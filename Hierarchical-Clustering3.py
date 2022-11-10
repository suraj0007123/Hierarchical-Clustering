## Importing data 
import pandas as pd

### Read the data into the python
telecomdata=pd.read_excel(r"E:\DESKTOPFILES\suraj\assigments\hierarchical clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")

telecomdata.columns ### To know the names of each column
telecomdata.shape  ## To know the shape of the data 
telecomdata.dtypes  ## To know the datatype of each variable

# Dropping the columns which unneccessary in the dataframe
telecomdata1=telecomdata.drop(['Count','Quarter','Referred a Friend','Number of Referrals','Tenure in Months','Contract','Paperless Billing','Payment Method'], axis=1)
telecomdata1.columns
telecomdata1.dtypes

## Changing the names of numerical variables (Rename)
telecomdata1.rename(columns={"Avg Monthly Long Distance Charges":"avgmonthlyloongdistancecharges","Avg Monthly GB Download":"avgmonthlygbdownload","Monthly Charge":"monthlycharge","Total Charges":"totalcharges","Total Refunds":"totalrefunds","Total Extra Data Charges":"totalextradatacharges","Total Long Distance Charges":"totallongdistancecharges","Total Revenue":"totalrevenue"},inplace=True)

telecomdata1.columns

### Identify duplicates records in the data ###
duplicate=telecomdata1.duplicated()
duplicate
sum(duplicate)

############## Outlier Treatment ###############
import seaborn as sns
sns.boxplot(telecomdata1.avgmonthlyloongdistancecharges)

telecomdata1.columns
sns.boxplot(telecomdata1.avgmonthlygbdownload)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail="both",
                          fold=1.5,
                          variables=['avgmonthlygbdownload'])
df_t=winsor.fit_transform(telecomdata1[['avgmonthlygbdownload']])
sns.boxplot(df_t.avgmonthlygbdownload)

telecomdata1.columns

sns.boxplot(telecomdata1.monthlycharge)

telecomdata1.columns
sns.boxplot(telecomdata1.totalcharges)

telecomdata1.columns
sns.boxplot(telecomdata1.totalrefunds)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['totalrefunds'])
df_t=winsor.fit_transform(telecomdata1[['totalrefunds']])
sns.boxplot(df_t.totalrefunds)

telecomdata1.columns

sns.boxplot(telecomdata1.totalextradatacharges)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['totalextradatacharges'])
df_t=winsor.fit_transform(telecomdata1[['totalextradatacharges']])
sns.boxplot(df_t.totalextradatacharges)

telecomdata1.columns

sns.boxplot(telecomdata1.totallongdistancecharges)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['totallongdistancecharges'])
df_t=winsor.fit_transform(telecomdata1[['totallongdistancecharges']])
sns.boxplot(df_t.totallongdistancecharges)

telecomdata1.columns

sns.boxplot(telecomdata1.totalrevenue)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['totalrevenue'])
df_t=winsor.fit_transform(telecomdata1[['totalrevenue']])
sns.boxplot(df_t.totalrevenue)

#### zero variance and near zero variance ######
telecomdata1.var()

telecomdata1.isna().sum()  # check for count of NA'sin each column

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

# Normalized data frame (considering the numerical part of data)
telecomdata2_norm=norm_func(telecomdata1.iloc[:,16:])
telecomdata2_norm


telecomdata1.shape
telecomdata1.columns

telecomdata1.dtypes


telecommdata3_norm = norm_func(telecomdata1.iloc[:, telecomdata1.columns.get_loc('avgmonthlyloongdistancecharges')])
telecommdata4_norm = norm_func(telecomdata1.iloc[:, telecomdata1.columns.get_loc('avgmonthlygbdownload')])

telecomm_new = pd.concat([telecomdata2_norm, telecommdata3_norm,telecommdata4_norm], axis =1)
telecomm_new.columns
telecomm_new.shape
telec=telecomdata1[['Offer', 'Phone Service','Multiple Lines','Internet Service','Internet Type','Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data']]
telec.dtypes

tel=pd.concat([telec, telecomm_new ], axis =1)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(telecomm_new, method="complete", metric="euclidean")

import matplotlib.pyplot as plt
# Dendrogram
plt.figure(figsize=(15,8));plt.title("Hierarchical Clustering Dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
       leaf_rotation=0,   # rotates the x axis labels
       leaf_font_size=8, # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=6, linkage="complete", affinity="euclidean").fit(telecomm_new)
h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_) 
telecomm_new.head()
telecomm_new.shape

telecomm_new['clust']=cluster_labels   # creating a new column and assigning it to new column 
telecomm_new.head()
telecomm_new.shape

# Aggregate mean of each cluster
telecomm_mean=telecomdata1.iloc[:,].groupby(telecomm_new.clust).mean()
telecomm_mean.shape
telecomm_mean.columns
telecomm_mean.head
