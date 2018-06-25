# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:12:49 2018

@author: Lenovo
"""

import pandas as pd
data=pd.read_csv("affairs.csv")
features=data.iloc[:,0:-1].values
labels=data.iloc[:,-1].values





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





#optimize the solution
import statsmodels.formula.api as sm
import numpy as np
features=np.append(arr=np.ones((6366,1)).astype(int),values=features,axis=1)
features_opt=features[:,[0,1,2,3,4,5,6,7,8]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()

features_opt=features[:,[0,1,2,3,4,6,7,8]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()


features_opt=features[:,[0,1,2,3,4,7,8]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()

features_opt=features[:,[0,1,2,3,4,8]]
reg_ols=sm.OLS(endog=labels,exog=features_opt).fit()
reg_ols.summary()

coef=reg_ols.params