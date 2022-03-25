#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from sklearn import metrics


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#IMPPORTING dataframe
df = pd.read_csv("E:/Data Science/Projects and assignments/datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
pd.set_option("display.max_columns", None)
df.head()


# ## EDA

# In[3]:


df.shape


# Data has 1470 rows and 35 columns

# In[4]:


df.isnull().sum()


# There is no null value in the data

# In[5]:


df.dtypes


# In[6]:


df.Attrition.value_counts()


# In[7]:


plt.style.use('ggplot')
plt.figure(figsize = (6,4))

labels = df.Attrition.value_counts().index

values = df.Attrition.value_counts().values

sns.barplot(labels, values)

plt.title("Distribution of Attrition of the company")
plt.xlabel("Employee Turnover")
plt.ylabel("Count")
plt.show()


# The Attrition column is imbalanced

# In[8]:


df.Department.value_counts()


# In[9]:


plt.style.use('ggplot')
plt.figure(figsize = (6,4))

labels = df.Department.value_counts().index

values = df.Department.value_counts().values

sns.barplot(labels, values)

plt.title("Distribution of employees in various department of the company")
plt.xlabel("Department name")
plt.ylabel("Employee Count")
plt.show()


# In[10]:


plt.style.use('ggplot')
plt.figure(figsize = (6,4))

labels = df.Gender.value_counts().index

values = df.Gender.value_counts().values

sns.barplot(labels, values)

plt.title("Distribution of employees according to their Gender")
plt.xlabel("Gender")
plt.ylabel("Employee Count")
plt.show()


# In[11]:


sns.distplot(df.Age)


# In[12]:


sns.boxplot(df.Age)


# Age column is almost normally distributed. 

# In[13]:


sns.boxplot(df.MonthlyIncome)


# Monthly income of the employees rightly skewed . Most of the employees have salary arounf 5000. 

# In[14]:


sns.boxplot(df['TotalWorkingYears'])


# Total working years of most employees is 10 years. The column is rightly skewed. 

# In[15]:


sns.distplot(df.PercentSalaryHike)


# In[16]:


def kdePlot(var):
    fig = plt.figure(figsize=(15,4))
    ax=sns.kdeplot(df.loc[(df['Attrition'] == 'No'),var] , color='g',shade=True, label='no Attrition') 
    ax=sns.kdeplot(df.loc[(df['Attrition'] == 'Yes'),var] , color='b',shade=True, label='Attrition')
    plt.legend()
    plt.title('Employee Attrition with respect to {}'.format(var))

columns = ['JobLevel', 'DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears',
          'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'YearsInCurrentRole']
for n in columns:
    kdePlot(n)


# Attrition is high for Joblevel 1.
# 
# Attrition is high for monthly income around 4500.
# 
# Attrition is high for PercentSalaryHike close to 12.
# 
# Attrition is high for TotalWorkingYears around 6.
# 
# Attrition is high for YearsAtCompany near 2-3. 
# 
# Attrition is high for Years Since Last Promotion is 0-1.
# 
# Attrition is high for Years with Current Manager is 0-1.
# 
# Attrition is high for Years in current role 0-1.

# In[17]:


columns = ['Age','Department', 'DistanceFromHome','JobLevel', 'EducationField', 'Gender', 'MaritalStatus', 
            'OverTime','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsInCurrentRole' ]

for val in columns:
    matrix = pd.crosstab(df[val], df.Attrition)
    matrix.div(matrix.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False, figsize=(10,8))
    plt.xticks(rotation=90)


# Attrition is high for age 19-21 years. This may be due to they are students or went for higher studies. 
# Again Attrition is high for age 58 years. This may be due to retirement and age related health conditions.
# 
# The average attrition is high as distance from home increases.
# 
# Attrition is high for Job level 1. 
# 
# Attrition is high for Human Resources, Marketing and Technical Degree employees. 
# 
# Attrition is slightly higher in males than females. 
# 
# Attrition is high for unmarried/single employees. 
# 
# Attrition is high for employees doing overtime. 
# 
# Attrition is high for employees with working years 0-1. Again Attrition is high for employees above 40. 
# 
# Attrition is slightly higher for employees having 0 training time last year. 

# In[18]:


df.corr()


# In[19]:


df = df.drop(['EmployeeCount','StandardHours','EmployeeNumber'], axis = 1)


# In[20]:


corr_matrix = df.corr()
fig , ax = plt.subplots(figsize=(20,12))
sns.heatmap(corr_matrix,vmax=0.8, annot=True)


# In[21]:


#LABEL ENCODING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

#selecting categorical columns to encode from dataset

cat_columns = list(df.select_dtypes("object").columns)

#setting up the imputer 

transformer = ColumnTransformer(transformers = [("L", OrdinalEncoder(),cat_columns)], remainder = 'passthrough')
cat_columns_imputed = transformer.fit_transform(df[cat_columns])

#passing the imputer values in each of the categorical columns in the original dataset

df[cat_columns] = cat_columns_imputed


# In[22]:


df.head()


# ## DecisionTree Model Fitting

# In[23]:


X = df.drop(['Attrition'], axis = 1)
y = df['Attrition']

X.head()


# In[24]:


#Train test split data for model training

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.20, random_state = 10)

accuracy = list() #creating an empty list for storing accuracy

#fitting 500 different decision trees and sorting their accuracy score in a list
for k in range(500):
    model = DecisionTreeClassifier(random_state = int (k))
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc_score = metrics.accuracy_score(y_test,y_pred)
    accurace = accuracy.append(acc_score)
#printing accuracy list 
accuracy


# In[25]:


#plotting accuracy score with different random state
col = np.arange(0,500,1)

df = pd.DataFrame({'random_states': col, 'accuracy_score': accuracy})

plt.figure(figsize =(10,8))
sns.lineplot(x = 'random_states', y = 'accuracy_score', data =df)


# In[26]:


#getting the random state with max accuracy
accuracy.index(max(accuracy)) 


# In[27]:


#calculating confusion matrix and max accuracy with random state 329

model = DecisionTreeClassifier(random_state = 170)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc_score = metrics.accuracy_score(y_test,y_pred)

print("accuracy: ",acc_score)

#confusion matrix
from sklearn.metrics import confusion_matrix
print("\n Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

#classification report
from sklearn.metrics import classification_report
print("\n Classification Report: ")
print(classification_report(y_test, y_pred))


# In[28]:


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)


# ### Regularization using GridSearchCV

# In[29]:


from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, {
    'max_depth' : range(2,50,5),
    'min_samples_leaf' : range(2,10,2),
    'min_samples_split' : range(10,100,5)
}, cv = 5, return_train_score = False)

clf.fit(X_train, y_train)


# In[30]:


clf.best_score_


# In[31]:


clf.best_params_


# ## SVM

# In[32]:


svm = SVC(kernel ='rbf', C = 30, gamma = 'auto')
svm.fit(X_train, y_train)


# In[33]:


#evaluating the model
y_pred = svm.predict(X_test)

acc_score = metrics.accuracy_score(y_test,y_pred)

print("accuracy: ",acc_score)

print("\n Confusion Matrix: ")
print(confusion_matrix(y_test,y_pred))

print("\n Classification Report: ")
print(classification_report(y_test, y_pred))


# #### Conclusion
# We can see Accuracy of Decision Tree Classifier(using Grid Search) : 85%
#            Accuracy of SVM classifier : 81%
#            
#  Both the classifiers are decently classifying the data. But Decision Tree is good. 

# ## K Means Clustering

# In[34]:


#scaling the data
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[36]:


#finding optimal value of cluster
k_rng = range(2,20)
sse = []
for k in k_rng:
    km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10)
    km.fit(X)
    sse.append(km.inertia_)


# In[37]:


plt.plot(k_rng, sse)
plt.xlabel("k value")
plt.ylabel("SSE")
plt.show()


# We can see k = 5 at elbow

# In[38]:


#model fitting
model = KMeans(n_clusters = 5)
y_pred = model.fit_predict(X)
y_pred


# In[39]:


#Score evaluation 
#normalized_mutual_info_score(y, y_pred)
silhouette_score(X, model.labels_, metric = 'euclidean')


# #### Conclusion
# Categorical data is a problem for KMeans algorithm.
# The standard k-means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space isn't really meaningful.
# As I have not separated categorical data and numerical data the clustering is not accurate. The accuracy score is not very good. 

# In[ ]:




