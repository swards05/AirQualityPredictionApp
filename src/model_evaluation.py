#src/model_evaluation.py

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test, model_type='MLP'):
    if model_type == 'LSTM':
        X_test = np.expand_dims(X_test, axis=1)
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'MAE': mae, 'MSE': mse, 'R2': r2}
