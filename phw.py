
from math import gamma
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.utils import shuffle



# PHW#1 function
# Problem: Compare performance (accuracy) of the following classification models against the same dataset.
#  - This function will try combinations of the various models automatically.
#  - This function let us know what scaler, model, and hyperparameter has the best score.
#  - This function was documented by pydoc.
def findBestOptions(
    X:DataFrame,
    y:DataFrame,
    scalers=[StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()],
    models=[
        DecisionTreeClassifier(criterion="gini"), DecisionTreeClassifier(criterion="entropy"), 
        LogisticRegression(solver='lbfgs'), LogisticRegression(solver='newton-cg'), LogisticRegression(solver='liblinear'), LogisticRegression(solver='sag'), LogisticRegression(solver='saga'), 
        SVC(kernel='rbf',probability=True),SVC(kernel='rbf', gamma = 0.001,probability=True),SVC(kernel='rbf', gamma = 0.01,probability=True),SVC(kernel='rbf', gamma = 0.1,probability=True),SVC(kernel='rbf', gamma = 1,probability=True),SVC(kernel='rbf', gamma = 10,probability=True),
        SVC(kernel='poly',probability=True),SVC(kernel='poly', gamma = 0.001,probability=True),SVC(kernel='poly', gamma = 0.01,probability=True),SVC(kernel='poly', gamma = 0.1,probability=True),SVC(kernel='poly', gamma = 1,probability=True),SVC(kernel='poly', gamma = 10,probability=True),
        SVC(kernel='sigmoid',probability=True),SVC(kernel='sigmoid', gamma = 0.001,probability=True),SVC(kernel='sigmoid', gamma = 0.01,probability=True),SVC(kernel='sigmoid', gamma = 0.1,probability=True),SVC(kernel='sigmoid', gamma = 1,probability=True),SVC(kernel='sigmoid', gamma = 10,probability=True),
        SVC(kernel='linear',probability=True),SVC(kernel='linear', gamma = 0.001,probability=True),SVC(kernel='linear', gamma = 0.01,probability=True),SVC(kernel='linear', gamma = 0.1,probability=True),SVC(kernel='linear', gamma = 1,probability=True),SVC(kernel='linear', gamma = 10,probability=True),
    ],
    cv_k=[2,3,4,5,6,7,8,9,10],
    isCVShuffle = True,
):
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
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

    Returns
    ----------
    - `best_params_`: dictionary
      - `best_scaler_`: Scaler what has best score.
      - `best_model_`: Model what has best score.
      - `best_cv_k_`: k value in K-fold CV what has best score.
    - `best_score_`: double
      - Represent the score of the `best_params`.

    See Also
    ----------
    to_csv : Write DataFrame to a comma-separated values (csv) file.

    Examples
    ----------
    >>> pd.read_csv('data.csv')  # doctest: +SKIP
    """

    # Initialize variables
    maxScore = -1.0
    best_scaler = None
    best_model = None
    best_cv_k_ = None

    # Find best scaler
    for n in range(0, len(scalers)):
        X = scalers[n].fit_transform(X)

        # Find best model
        for m in range(0, len(models)):
            # Find best k value of CV
            for i in range(0, len(cv_k)):
                kfold = KFold(n_splits=cv_k[i], shuffle=isCVShuffle)
                score_result = cross_val_score(models[m], X, y, cv=kfold)
                # if mean value of scores are bigger than max variable,
                # update new options(model, scaler, k) to best options
                if maxScore < score_result.mean():
                    maxScore = score_result.mean()
                    best_scaler = scalers[n]
                    best_model = models[m]
                    best_cv_k_ = cv_k[i]

    # Return value with dictionary type
    return {
        'best_params_': {
            'best_scaler_': best_scaler,
            'best_model_' : best_model,
            'best_cv_k_': best_cv_k_,
        },
        'best_score_': maxScore
    }



# Plot ROC Curve
def plot_roc_curve(X, y, model, title):
    # Calculate False Positive Rate, True Positive Rate
    prob = model.predict_proba(X)
    fpr, tpr, _ = roc_curve(y, prob[:, 1])

    # Plot result
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, color='b', label='ROC')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for ' + str(title), fontsize=20)

    plt.legend()
    plt.show()

# Plot Heatmap
def heatmap(X, title):
    # Calculate correlation matrix and plot them
    plt.figure(figsize=(12,10))
    plt.title('Heatmap of ' + str(title), fontsize=20)
    g=sns.heatmap(X[X.corr().index].corr(), annot=True, cmap="YlGnBu")

    plt.show()



### Source code
# Import dataset
df = pd.read_csv("./data.csv")

## Preprocessing
# Drop useless feature
df.drop(["Sample code number"], axis=1, inplace=True)

# Drop missing value
df.drop( df[ (df['Bare Nuclei'] == '?')].index, inplace=True)

# Change target(Class) value (2 -> 0 / 4 -> 1)
df.at[df[df['Class (2, 4)'] == 2].index, 'Class (2, 4)'] = 0
df.at[df[df['Class (2, 4)'] == 4].index, 'Class (2, 4)'] = 1

# Split feature and target data
X = pd.DataFrame(df.iloc[:,0:9], dtype=np.dtype("int64"))
y = df.iloc[:,9]

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

## Visualization

# Make dataframe to plot heatmap
dft = pd.DataFrame(X, columns=columns)
dft['target'] = y

# Show heatmap
title = str(best_scaler) + " / " + str(best_model)
heatmap(dft, title)

# Show ROC Curve
plot_roc_curve(test_X, test_y, model, title)