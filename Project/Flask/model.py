import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# import data
data = pd.read_csv("smoke_detection_iot.csv", encoding="latin1")
data.head()

# Feature Extraction
data.drop(columns=['Unnamed: 0', 'UTC', 'CNT'], inplace=True)
cols = [1, 3, 6, 7, 8, 9, 10, 12]
data_1 = data[data.columns[cols]]
data = data_1

# Splitting data
X = data.iloc[:, :-1]
y = data['Fire Alarm']

# Oversampling
ros = RandomOverSampler(sampling_strategy=0.65)
X_bal, y_bal = ros.fit_resample(X, y)

print(y.value_counts())
print(y_bal.value_counts())

# Splitting into Training and testing
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.15, random_state=39)
print(X_train.head())

# model building
knn = KNeighborsClassifier(n_neighbors=850)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# model deployment
pickle.dump(knn, open('model.pkl', 'wb'))
print(accuracy_score(y_test, y_pred))
