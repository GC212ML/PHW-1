# Programming Homework #1
 - Auto ML for classification

## findBestClassifierOptions
 - Goal: Compare performance (accuracy) of the following classification models against the same dataset.
    - This function will try combinations of the various models automatically.
    - This function let us know what scaler, model, and hyperparameter has the best score.
    - This function was documented by pydoc.

### Parameter
- `X`: pandas.DataFrame
    - training dataset.
- `y`: pandas.DataFrame
    - target value.
- `scalers`: array
    - Scaler functions to scale data. This can be modified by user.
    - StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler as default.
- `models`: array
    - Model functions to fitting data and prediction. This can be modified by user.
    - DecisionTreeClassifier, LogisticRegression, SVC as default with hyperparameters.
- `k`: array
    - Cross validation parameter. Default value is [2,3,4,5,6,7,8,9,10].

### Returns
- `best_params_`: dictionary
    - `best_scaler_`: Scaler what has best score.
    - `best_model_`: Model what has best score.
    - `best_cv_k_`: k value in K-fold CV what has best score.
- `best_score_`: double
    - Represent the score of the `best_params`.


### Example
```python
## Find Best model and options
# Run findBestOptions()
result = findBestOptions(X, y)

# Extract results
best_score = result['best_score_']
result = result['best_params_']
best_scaler = result['best_scaler_']
best_model = result['best_model_']

# Print the result of best option
print("\nBest Scaler: ", end="")
print(best_scaler)
print("Best Model: ", end="")
print(best_model)
print("Score: ", end="")
print(best_score)
print("")

# Fit model with best options
columns = X.columns
X = best_scaler.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, shuffle=True)
model = best_model.fit(train_X, train_y)
print("Model score: ", end="")
print(model.score(test_X, test_y))
```