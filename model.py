# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('Cleaned_MCS.csv')

X = df.drop('Compression_Strength', axis=1)
y = df['Compression_Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

# Random Forest Regressor
model_rf = RandomForestRegressor(random_state=50)

#Fitting model with trainig data
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
}
grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

best_rf.fit(X_train, y_train)

# Saving model to disk
pickle.dump(best_rf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[486.42, 180.60, 21.26, 201.66, 16.11,1151.17,708.50,344.43]]))