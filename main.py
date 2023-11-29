from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor


random_grid = {
  'n_estimators': [5,20,50,100],
  'max_features': ['auto', 'sqrt'],
  'max_depth': [int(x) for x in np.linspace(10, 120, num = 12)],
  'min_samples_split': [2, 6, 10],
  'min_samples_leaf': [1, 3, 4],
  'bootstrap': [True, False]
}
