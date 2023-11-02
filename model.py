# Importing the libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle

df = pd.read_csv('Cleaned_MCS.csv')

X = df.drop('Compression_Strength', axis=1)
y = df['Compression_Strength']

#Splitting Training and Test Set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

param_grid = {
    'n_estimators': [50, 100, 150],  # You can adjust the number of trees
    'max_depth': [None, 10, 20, 30],  # You can adjust the maximum depth of trees
    'min_samples_split': [2, 5, 10],  # You can adjust the minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # You can adjust the minimum samples required for a leaf node
}

rf_model = RandomForestRegressor(random_state=42)

# Create a Grid Search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the Grid Search to find the best hyperparameters
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_rf_model = RandomForestRegressor(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)


# Saving model to disk
pickle.dump(best_rf_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6, 8, 9, 7, 8, 7]]))