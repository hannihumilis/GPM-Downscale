from sklearn.ensemble import RandomForestRegressor
import functions
random_grid = {
  'n_estimators': [5,20,50,100],
  'max_features': ['auto', 'sqrt'],
  'max_depth': [int(x) for x in np.linspace(10, 120, num = 12)],
  'min_samples_split': [2, 6, 10],
  'min_samples_leaf': [1, 3, 4],
  'bootstrap': [True, False]
}

model = RandomForestRegressor(random_state=42)
print(functions.cv_downscaling(model, random_grid))
