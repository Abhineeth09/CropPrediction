import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

pd.set_option("display.max_rows",100)
pd.set_option("display.max_columns",100)
pd.set_option("display.max_colwidth",100)
pd.set_option("display.width",175)
#predicting the best crop for the features given
path=r'/Users/abhineethmishra/Downloads/convertcsv.csv'
df=pd.read_csv(path)
#drop column with missing values
df=df.drop(columns=['FIELD1'])
df=df.drop(columns=['S_UNIT'])
#creating a feature set my removing unnecassary features
Feature_set=df.copy()
Feature_set=Feature_set.drop(columns=['S_LAND_ID','S_LAND_TYPE','S_USERID','_id'])

output=Feature_set.S_TREE_CROP_NAME
Feature_set=Feature_set.drop(columns=['S_TREE_CROP_NAME'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#print(output)
#print(Feature_set)
clf = RandomForestClassifier()

print(Feature_set)
#Encoding to convert String attributes into numbers
from sklearn.preprocessing import LabelBinarizer
croptype_lb = LabelBinarizer()
soiltype_lb = LabelBinarizer()
irrigation_lb=LabelBinarizer()
croptype = croptype_lb.fit_transform(Feature_set.S_TREE_CROP_TYPE.values)
soiltype = soiltype_lb.fit_transform(Feature_set.S_SOIL_TYPE.values)
irrigation = irrigation_lb.fit_transform(Feature_set.S_IRRIGATION.values)

temp=Feature_set.copy()
temp=temp.drop(columns=['S_IRRIGATION','S_SOIL_TYPE','S_TREE_CROP_TYPE'])
temp['Crop Type']=pd.DataFrame(croptype)
temp['Irrigation']=pd.DataFrame(irrigation)
stype=pd.DataFrame(soiltype)
temp["Soil Type"]=(stype[0].astype(str)+stype[1].astype(str)+stype[2].astype(str)+stype[3].astype(str)+stype[4].astype(str)+stype[5].astype(str)).astype(int)
#Final Feature set is in temp
Final_Feature_Set=temp.copy()
#print(Final_Feature_Set)
#print("Correlations")
#print(Final_Feature_Set.corr())
Y=output
#dividing into train and test set
xTrain, xTest, yTrain, yTest = train_test_split(Final_Feature_Set, Y, test_size = 0.2, random_state = 1)


#print(Y)
clf.fit(xTrain, yTrain)
#Test output
#print(clf.predict([[3487,0,0.00,1,0,1000]]))
#find out accuracy

score = accuracy_score(yTest,clf.predict(xTest))
print("Accuracy on Test Set:"+score.astype(str))
from sklearn.cross_validation import cross_val_score
cross_val=np.mean(cross_val_score(clf, xTrain, yTrain, cv=10)).astype(str)
print("Cross Validation Accuracy"+cross_val)

grid_param = {  
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


from sklearn.model_selection import GridSearchCV
#Finding the best parameters for grid search
#gd_sr = GridSearchCV(estimator=clf,  
#                     param_grid=grid_param,
#                     scoring='accuracy',
#                     cv=5,
#                     n_jobs=-1)
#gd_sr.fit(xTrain, yTrain)  
#best_parameters = gd_sr.best_params_ 
best_parameters={'bootstrap': True, 'criterion': 'entropy', 'n_estimators': 100} 
print('best parameters:')
print(best_parameters)


#using the best parameters
#best_result = gd_sr.best_score_  
#print('Best Result')
#print(best_result)
#using best parameters as best parameters:
#{'bootstrap': True, 'criterion': 'gini', 'n_estimators': 100}
Updatedclf = RandomForestClassifier(**best_parameters)
Updatedclf.fit(xTrain, yTrain)
print(Updatedclf.predict([[100,62,29.63,0,0,0]]))
