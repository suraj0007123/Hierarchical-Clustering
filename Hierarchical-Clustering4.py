### Importing the data
import pandas as pd
##### Reading the data into python
insurance=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\hierarchical clustering\Dataset_Assignment Clustering\AutoInsurance.csv")

insurance.info() ### To know the info about the dataframe
insurance.describe() ## To know the stats about the dataframe
insurance.dtypes ### To know the datatype of each variable

############## Outlier Treatment ###############
import seaborn as sns

insurance.rename(columns={"Customer Lifetime Value":"customerlifetimevalue","Monthly Premium Auto":"monthlypremiumauto","Months Since Last Claim":"monthssincelastclaim","Months Since Policy Inception":"monthssincepolicyinception","Number of Open Complaints":"numberofopencomplaints","Number of Policies":"numberofpolicies","Total Claim Amount":"totalclaimamount"},inplace=True)

insurance.columns

sns.boxplot(insurance.customerlifetimevalue)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['customerlifetimevalue'])
df_t=winsor.fit_transform(insurance[['customerlifetimevalue']])
sns.boxplot(df_t.customerlifetimevalue)

insurance.columns

sns.boxplot(insurance.Income)

sns.boxplot(insurance.monthlypremiumauto)
from feature_engine.outliers Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['monthlypremiumauto'])
df_t=winsor.fit_transform(insurance[['monthlypremiumauto']])
sns.boxplot(df_t.monthlypremiumauto)

insurance.columns
sns.boxplot(insurance.monthssincelastclaim)

insurance.columns

sns.boxplot(insurance.monthssincepolicyinception)

insurance.columns

sns.boxplot(insurance.numberofopencomplaints)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['numberofopencomplaints'])
df_t=winsor.fit_transform(insurance[['numberofopencomplaints']])
sns.boxplot(df_t.numberofopencomplaints)

insurance.columns

sns.boxplot(insurance.numberofpolicies)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['numberofpolicies'])
df_t=winsor.fit_transform(insurance[['numberofpolicies']])
sns.boxplot(df_t.numberofpolicies)

insurance.columns

sns.boxplot(insurance.totalclaimamount)
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                          tail='both',
                          fold=1.5,
                          variables=['totalclaimamount'])
df_t=winsor.fit_transform(insurance[['totalclaimamount']])
sns.boxplot(df_t.totalclaimamount)


### Identify duplicates records in the data ###
duplicate=insurance.duplicated()
duplicate
sum(duplicate)
insurance.shape

#take categorical data into one file for lable encoding
insurancedata=insurance[['Coverage','EmploymentStatus','Location Code','Policy Type','Policy','Renew Offer Type']]
insurancedata.columns

#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder 
# creating instance of labelencoder
labelencoder=LabelEncoder()

# Data Split into Input and Output variables
X=insurancedata.iloc[:,0:6]

y=insurancedata['Renew Offer Type']

insurancedata.columns

X['Coverage']=labelencoder.fit_transform(X['Coverage'])
X['EMplaymentStatus']=labelencoder.fit_transform(X['EmploymentStatus'])
X['Location Code']=labelencoder.fit_transform(X['Location Code'])
X['Policy Type']=labelencoder.fit_transform(X['Policy Type'])
X['Policy']=labelencoder.fit_transform(X["Policy"])
X['Renew Offer Type']=labelencoder.fit_transform(X['Renew Offer Type'])

### label encode y ###
y=labelencoder.fit_transform(y)
y=pd.DataFrame(y)
### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
insurance_new=pd.concat([X,y],axis=1)
insurance_new.columns

## rename column name
insurance_new=insurance_new.rename(columns={0:"Renew Offer Type"})

#take numerical data for normalization
insurance1=insurance[["Customer","customerlifetimevalue","Income","monthlypremiumauto","monthssincelastclaim","monthssincepolicyinception","numberofpolicies","totalclaimamount"]]
insurance1.describe() ## To know the stats of all the numerical columns
insurance1.info() # To know the info about the dataframe
insurance.columns

insurancedata1=insurance1.drop(['Customer'],axis=1)

# Normalization function 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
# Normalized data frame (considering the numerical part of data)
df_norm=norm_func(insurance1.iloc[:,1:])

df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

import matplotlib.pyplot as plt
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=5,linkage="complete",affinity="euclidean").fit(df_norm)
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)

insurance['clust']=cluster_labels # creating a new column and assigning it to new column 

insurance.head()
# Aggregate mean of each cluster
insurance.iloc[:,].groupby(insurance.clust).mean()

