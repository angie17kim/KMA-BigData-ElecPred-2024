import numpy as np
from scipy.stats import pearsonr

def FocusMSE(y_pred, dtrain, slope):
    y_true = dtrain.get_label()
    #weights = 1.0 + (slope - 1.0) * np.maximum(0, y_true)
    weights = np.exp(0.5 * slope * np.square(np.maximum(0, y_true)))
    
    grad = weights * (y_pred - y_true)
    hess = weights
    return grad, hess

def FocusEvalMetric(y_pred, dtrain, slope):
    y_true = dtrain.get_label()
    #weights = 1.0 + (slope - 1.0) * np.maximum(0, y_true)
    weights = np.exp(0.5 * slope * np.square(np.maximum(0, y_true)))
    
    weighted_rmse = np.sqrt(np.mean(weights * np.square(y_pred - y_true)))
    return 'focus_rmse', weighted_rmse

# 사용자 정의 평가 메트릭
def pearson_eval(y_val, y_pred, categories, return_groupwise=False):
    assert len(y_val) == len(y_pred)
    assert len(y_val) == len(categories)
    
    pearson_correlations = {}
    
    for category in np.unique(categories):
        category_mask = categories == category
        if np.sum(category_mask) > 1:
            pearson_correlation, _ = pearsonr(y_val[category_mask], y_pred[category_mask])
            pearson_correlations[category] = pearson_correlation
        else:
            raise ValueError('Category has one or no element')
    
    average_correlation = np.mean(list(pearson_correlations.values()))

    if return_groupwise:
        return average_correlation, pearson_correlations
    else:
        return average_correlation
