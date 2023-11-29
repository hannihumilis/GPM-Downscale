from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import functions
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class RandomForestModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y


        X_train, X_test, y_train,y_test , X_scaled_original = functions.prepare_data(X, y)

        random_grid = {
          'n_estimators': [5, 20, 50, 100],
          'max_features': ['sqrt', 'log2', None],  # Include 'log2' and None instead of 'auto'
          'max_depth': [int(x) for x in np.linspace(10, 120, num=12)],
          'min_samples_split': [2, 6, 10],
          'min_samples_leaf': [1, 3, 4],
          'bootstrap': [True, False]
        }
        # Assuming you have your training data X_train and corresponding labels y_train
        random_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=random_grid,
            n_iter=100,  # Number of random combinations to try
            scoring='neg_mean_squared_error',  # Choose an appropriate metric
            cv=5,  # Number of cross-validation folds
            verbose=2,  # Increase for more details
            random_state=42,
        n_jobs=-1  # Use all available processors
        )

        # Fit the random search to your data
        random_search.fit(X_train, y_train)

        # Access the best hyperparameters
        self.best_params = random_search.best_params_
        print("Best Hyperparameters:", self.best_params)

        # Access the best estimator (model)
        self.best_model = random_search.best_estimator_
        print("Best Model:", self.best_model)
        self.y_pred = self.best_model.predict(X_test)

        print("MSE:\t", mean_squared_error(y_test, self.y_pred))
        print("R^2:\t", r2_score(y_test, self.y_pred))
          #joblib.dump(best_model, '/Users/hanni/PycharmProjects/Sattelite/test.joblib')



