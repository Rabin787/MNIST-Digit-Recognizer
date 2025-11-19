# Finding the best fit model using SVM with hyperparameter tuning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

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


# Defining the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
#initializing the SVM classifier
svm_clf = SVC()
# Setting up GridSearchCV
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid,cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
# Fitting the model
grid_search.fit(x_train_scaled, y_train)
# Best parameters from grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
# Making predictions with the best estimator
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(x_test_scaled)
# Evaluating the best model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred_best))