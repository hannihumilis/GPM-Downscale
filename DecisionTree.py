from sklearn.tree import DecisionTreeRegressor
import functions
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20, 50, 100]
}
model = DecisionTreeRegressor(random_state=42)

print(functions.cv_downscaling(model, params))
