import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# Data set from Kaggle: 
    
def clean_dataset_column(features, col):
    for index in range(len(features)):
        if type(features.loc[index, col]) != 'str':
            features.loc[index, col] = str(features.loc[index, col])
    
    return features

def label_encode(features):        
    for column in features.columns:
        if features[column].dtype == type(object): 
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])

    return features

def one_hot_encoder(features):        
    for column in features.columns:
        if features[column].dtype == type(object): 
            ohe = OneHotEncoder()
            features[column] = ohe.fit_transform(features[column])

    return features

def view_skewness(feature):
    plt.figure(figsize=(25, 5))
    for idx, col in enumerate(feature.columns):
        plt.subplot(1,len(feature.columns),idx+1)
        plt.hist(train[col])
        plt.title(col)

def normalize_data(feature):
    for col in feature.columns:
        arr = np.asarray(feature[col].values)
        if 0 in arr:
            arr = arr + 1
        train_data,fitted_lambda = scipy.stats.boxcox(arr)
        print("col:", col, ":", skew(train_data))
        feature[col] = train_data
    return feature
        
def perform_data_validation(features):
    features = features.drop(columns=["PassengerId", "Cabin"])
    features = clean_dataset_column(features, "Embarked")
    features = features.fillna(features.mean())
    #features = label_encode(features)
    features = label_encode(features)
    return features

def create_linear_model(feature, label):
    linear_model = LinearRegression()
    training_label = feature.loc[:, label]
    linear_model.fit(feature.iloc[:, 1:], training_label)
    return linear_model

def create_logistic_model(feature, label):
    logistic_model = LogisticRegression()
    training_label = feature.loc[:, label]
    logistic_model.fit(feature.iloc[:, 1:], training_label)
    return logistic_model


def validate_model(model, test_features, valid_label, output_filename):
    test_label = list(model.predict(test_features))
    passenger_id = list(valid_label.PassengerId)
    output_ds = pd.DataFrame(data = list(zip(passenger_id,test_label)), 
                             columns= ["PassengerId", "Survived"])
    output_ds.to_csv(output_filename)
    return model.score(test_features, valid_label.Survived)

# if __name__ == "__main__":
path = os.getcwd()
train = pd.read_csv("creditcardfraud/train.csv")
test = pd.read_csv("creditcardfraud/test.csv")
valid_label = pd.read_csv("creditcardfraud/gender_submission.csv")
label = "Survived"

train = perform_data_validation(train)
print("TRAIN DATA")
view_skewness(train)
train.iloc[:, 1:] = normalize_data(train.iloc[:, 1:])
view_skewness(train)

linear_model = create_linear_model(train, label)
logistic_model = create_logistic_model(train, label)

test = perform_data_validation(test)
print("TEST DATA")
view_skewness(test)
test = normalize_data(test)
print("TEST DATA")
view_skewness(test)

print("Linear model Accuracy", validate_model(linear_model, test, valid_label, "linear.csv"))
print("Logistic model Accuracy", validate_model(logistic_model, test, valid_label, "logistic.csv"))
