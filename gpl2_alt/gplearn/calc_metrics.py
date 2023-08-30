from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.abs(mean_squared_error(y_true, y_pred)))
    target_range = np.max(y_true) - np.min(y_true)
    rmspe = (rmse / (0.5*target_range)) * 100
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, target_range, rmspe, r2
