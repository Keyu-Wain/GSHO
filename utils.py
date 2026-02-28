#  Utility functions
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics: R², RMSE, MSE, MAE"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"R²": r2, "RMSE": rmse, "MSE": mse, "MAE": mae}

def calculate_grid_size(param_grid):
    sizes = [len(v) for v in param_grid.values()]
    total = 1
    for s in sizes:
        total *= s
    return total