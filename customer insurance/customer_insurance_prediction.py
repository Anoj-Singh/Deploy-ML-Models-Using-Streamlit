import numpy as np
import pandas as pd
import pickle

df = pd.read_csv("D:\dataset\csv files\Social_Network_cls.csv")
# print(df.head())

x = df.drop('Purchased', axis=1)
y = df['Purchased']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(x_train, y_train)
# print(model.score(x_test, y_test))

pickle.dump(model, open('model.pkl', 'wb'))