#CropPrediction
This project aims to predict what kind of crop should be grown on some particular land, when some input such as Soil Type, Land Size, Moisture etc. are given.

Dependencies:
1.Python3
2.Pandas
3.Numpy
4.Sklearn
5.Matplotlib

Files included:
1.CSV File: Contains the dataset.
2.Python3 file: Contains the code.

Algorithm:
The algorithm used is a Random Forest Classifier. The dataset is divided into training and testing set (20% split). Best hyperparameters are found using GridSearchCV. This model is used to predict what crop is to be grown.