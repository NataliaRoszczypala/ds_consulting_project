import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

class RegressionAnalysis:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.predictions = {}

    def prepare_data(self):
        X = self.data.drop([self.target_column], axis=1)
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
    def fit_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['linear'] = model

    def tune_model(self, model, param_grid):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_, grid_search.best_params_
        
    def tune_ridge_regression(self, param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]}):
        model = Ridge()
        best_model, best_params = self.tune_model(model, param_grid)
        self.models['ridge'] = best_model
        return best_params

    def tune_lasso_regression(self, param_grid={'alpha': [0.1, 1.0, 10.0, 100.0]}):
        model = Lasso()
        best_model, best_params = self.tune_model(model, param_grid)
        self.models['lasso'] = best_model
        return best_params

    def tune_random_forest(self, param_grid={'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}):
        model = RandomForestRegressor(random_state=42)
        best_model, best_params = self.tune_model(model, param_grid)
        self.models['random_forest'] = best_model
        return best_params

    def tune_xgboost(self, param_grid={'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300]}):
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        best_model, best_params = self.tune_model(model, param_grid)
        self.models['xgboost'] = best_model
        return best_params
        
    def evaluate_models(self):
        evaluation = []
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            rmse = root_mean_squared_error(self.y_test, predictions)
            mape = mean_absolute_percentage_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            evaluation.append({
                'Model': name,
                'RMSE': rmse,
                'MAPE': mape,
                'R2': r2
            })
            self.predictions[name] = predictions
        return pd.DataFrame(evaluation)

    def get_coefficients(self, model_name):
        model = self.models[model_name]
        if hasattr(model, 'coef_'):
            return pd.DataFrame(model.coef_, self.X_train.columns, columns=['Coefficient'])
        elif hasattr(model, 'feature_importances_'):
            return pd.DataFrame(model.feature_importances_, self.X_train.columns, columns=['Importance'])
        else:
            raise ValueError(f"No coefficients or feature importances available for {model_name}")
