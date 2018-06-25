# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:00:54 2018

@author: Lenovo
"""

import pandas as pd
data=pd.read_csv("affairs.csv")
features=data.iloc[:,0:-1].values
labels=data.iloc[:,-1].values



#onehotencoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[6])
features=ohe.fit_transform(features).toarray()

#dummy variable trap
features=features[:,1:]

ohe=OneHotEncoder(categorical_features=[11])
features=ohe.fit_transform(features).toarray()

#dummy variable trap
features=features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)


#logistics regression
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(random_state=0)
lgr.fit(features_train,labels_train)

#prediction
labels_predict=lgr.predict(features_test)

#score
score=lgr.score(features_test,labels_test)

#making confussion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_predict)

#What percentage of total women actually had an affair
print(data["affair"].mean())

'''religious: women's rating of how religious she is (1 = not religious, 4 = strongly religious)

educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)

occupation: women's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)'''


'''Predict the probability of an affair for a random woman not present in the dataset. She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.'''


#prediction
labels_predict2=lgr.predict([1,0,0,0,0,0,0,1,0,0,1,25,3,1,4,16])


































