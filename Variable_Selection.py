#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[35]:


get_ipython().system('pip install Boruta')
get_ipython().system('pip install glmnet_py')


# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler


# ## Read data

# In[146]:


dataset = pd.read_csv('~/path/dataset.csv').dropna()
dataset.set_index("Date", inplace = True)
dataset


# In[3]:


dataset.info()


# ## Boruta algo

# In[1]:


# Separating the features from the target variable, and splitting the data into a train and a dev set
name = ""
X = dataset.drop(name, axis = 1)
y = dataset['Global crisis event 1D']



# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 1) 


# In[40]:


# Creating a baseline RandomForrestClassifier model with all the features

rf_all_features = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
rf_all_features.fit(X_train, y_train)

accuracy_score(y_test, rf_all_features.predict(X_test))


# In[41]:


# Creating a BorutaPy object with RandomForestClassifier as the estimator and ranking the features

rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2, random_state=1)
boruta_selector.fit(np.array(X_train), np.array(y_train))  


# In[42]:


print("Ranking: ",boruta_selector.ranking_)      
print("No. of significant features: ", boruta_selector.n_features_) 


# In[43]:


selected_rf_features = pd.DataFrame({'Feature':list(X_train.columns),'Ranking':boruta_selector.ranking_})
selected_rf_features.sort_values(by='Ranking') 





# In[44]:


X_important_train = boruta_selector.transform(np.array(X_train))
X_important_test = boruta_selector.transform(np.array(X_test)) 


# In[45]:


rf_boruta = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
rf_boruta.fit(X_important_train, y_train) 


# In[46]:


accuracy_score(y_test, rf_boruta.predict(X_important_test))




# ## Lasso 1

# In[162]:


features = dataset.iloc[:,0:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



# In[151]:


pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])


# In[152]:


search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )


# In[153]:


search.fit(X_train,y_train)


# In[154]:


search.best_params_



# In[155]:


coefficients = search.best_estimator_.named_steps['model'].coef_



# In[156]:


importance = np.abs(coefficients)



# In[157]:


np.array(features)[importance > 0]


# In[158]:


np.array(features)[importance == 0]


# ## Lasso 2

# In[171]:


Min_Max = MinMaxScaler()
X = Min_Max.fit_transform(X)
Y= Min_Max.fit_transform(y)


# In[170]:


# Split the data into 40% test and 60% training

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape


# In[166]:


# Selecting features using Lasso regularisation using SelectFromModel

sel_ = SelectFromModel(LogisticRegression(C=1, penalty=’l1', solver=’liblinear’))
sel_.fit(X_train, np.ravel(Y_train,order=’C’))
sel_.get_support()
X_train = pd.DataFrame(X_train)
                                          
                                          
                                          
                                          
                                          
                                          


# In[126]:


# See selected set of features

selceted_feat =  X_train.columns[(sel_.get_support())]
print(‘total features: {}’.format((X_train.shape[1])))
print(‘selected features: {}’.format(len(selected_feat)))
print(‘features with coefficients shrank to zero: {}’.format(
np.sum(sel_.estimator_.coef_ == 0)))




# In[127]:


# List of selected features

removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
removed_feats
X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
X_train_selected.shape, X_test_selected.shape




# In[ ]:


# To Check the Accuracy of the model we use Random Forest classifier to predict the results

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# Train the classifier
clf.fit(X_train_selected,np.ravel(Y_train,order=’C’))
# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test_selected)
# View The Accuracy Of Our Selected Feature Model
accuracy_score(Y_test, y_pred)




# In[ ]:




