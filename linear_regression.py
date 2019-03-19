import numpy as np

def linear_regression(covariates, targets):
    covariates = np.array(covariates)
    targets = np.array(targets).reshape(-1,1)
    combination = np.hstack([covariates, targets])
    covariates = covariates[~np.isnan(combination).any(axis=1)]
    targets = targets[~np.isnan(combination).any(axis=1)]
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(covariates), covariates)), np.transpose(covariates)), targets)
    y_hat = np.matmul(covariates, beta)
    errors = targets - y_hat
    variance = float(np.matmul(np.transpose(errors), errors) / (covariates.shape[0] - covariates.shape[1] - 1))
    variance_beta = []
    for index, coeff in enumerate(beta.tolist()):
        mean = float(covariates[:, index].mean())
        summation = 0
        for point in covariates[:, index].tolist():
            summation += pow(float(point) - mean, 2)
        variance_beta.append(variance/summation)
    variance_beta = np.array(variance_beta).reshape(-1,1)
    se_beta = np.sqrt(variance_beta)
    lower_bounds = beta - (2 * se_beta)
    upper_bounds = beta + (2 * se_beta)
    return beta[:,0], se_beta[:,0], lower_bounds[:,0], upper_bounds[:,0]