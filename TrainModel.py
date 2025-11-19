# Train the  model using the best fit parameters found in FindBestFit.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data=pd.read_csv("digit-recognizer//train.csv")
# Separating features and target variable
x=data.drop(columns="label")
y=data["label"]
# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# Initializing the SVM classifier with best parameters found
best_params = {
    'C': 10,
      'gamma': 'scale', 
      'kernel': 'poly'
}
svm_clf = SVC(**best_params)

# Fitting the model
svm_clf.fit(x_train_scaled, y_train)

# Making predictions
y_pred = svm_clf.predict(x_test_scaled)

# Saving the trained model
import joblib
joblib.dump(svm_clf, 'Model.joblib')
joblib.dump(scaler, 'scaler.joblib')
