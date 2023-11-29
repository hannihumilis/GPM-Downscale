from sklearn.svm import SVR
import functions
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
model = SVR()

print(functions.cv_downscaling(model, param_grid))
