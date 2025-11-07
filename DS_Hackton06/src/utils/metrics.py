from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
def binary_metrics(y_true, y_pred):
    return {'accuracy': accuracy_score(y_true,y_pred), 'f1': f1_score(y_true,y_pred)}
def regression_metrics(y_true, y_pred):
    return {'rmse': mean_squared_error(y_true,y_pred, squared=False)}
