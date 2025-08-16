import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class DataPreprocessing:
    def __init__(self, data_path):
        self.ytest = None
        self.ytrain = None
        self.xtest = None
        self.xtrain = None
        self.data_path = data_path
        self.data = None
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        return self.data
    def preprocess_data(self):
        self.data = self.data.drop('customerID', axis=1)
        self.data['TotalCharges'] = self.data['TotalCharges'].replace(' ', np.nan).astype(float)
        self.data.dropna(axis=0, inplace=True)
        categorical_features = self.data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for feature in categorical_features:
            self.data[feature] = le.fit_transform(self.data[feature])
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(X, y, test_size=0.2, random_state=42)
        return self.xtrain, self.xtest, self.ytrain, self.ytest
    def feature_selection(self):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.xtrain, self.ytrain)
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        threshold = 0.03
        importances_features = [i for i in range(len(importances)) if importances[i] >= threshold]
        print(f"Features with importance greater than {threshold} :")
        for i in importances_features:
            print(f"{self.xtrain.columns[i]} : {importances[i]}")
        importance_features = self.xtrain.columns[importances_features]
        self.xtrain = self.xtrain[importance_features]
        self.xtest = self.xtest[importance_features]

    def scale_data(self):
        numerical_features = self.xtrain.select_dtypes(include=['int64', 'float64']).columns.tolist()
        scaler = MinMaxScaler()
        self.xtrain[numerical_features] = scaler.fit_transform(self.xtrain[numerical_features])
        self.xtest[numerical_features] = scaler.transform(self.xtest[numerical_features])

    def save_data(self, xtrain, xtest, ytrain, ytest):
        data_train = pd.concat([xtrain, ytrain], axis=1)
        data_test = pd.concat([xtest, ytest], axis=1)
        if not os.path.exists("../data/processed"):
            os.makedirs("../data/processed")
        data_train.to_csv("../data/processed/train.csv", index=False)
        data_test.to_csv("../data/processed/test.csv", index=False)
        return data_train, data_test

if __name__ == "__main__":
    data_path = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    data_preprocessing = DataPreprocessing(data_path)
    data_preprocessing.load_data()
    data_preprocessing.preprocess_data()
    data_preprocessing.feature_selection()
    data_preprocessing.scale_data()
    data_train, data_test = data_preprocessing.save_data(data_preprocessing.xtrain, data_preprocessing.xtest, data_preprocessing.ytrain, data_preprocessing.ytest)