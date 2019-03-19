from linear_regression import linear_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.DataFrame.from_dict({"col1": [1,2,3,4,5,6,7,8,9,10],
                             "col2": [np.NaN,11,12,13,14,15,16,17,18,19],
                             "col3": [20,21,22,23,24,25,26,27,28,np.NaN],
                             "response": [29,30,31,np.NaN,33,34,35,36,37,38]})

covariates = df.drop("response", axis=1).values
targets = df["response"].values

beta, se_beta, lower_bounds, upper_bounds = linear_regression(covariates, targets)

result_table = pd.DataFrame.from_dict({"lower_bound_for_estimates": lower_bounds,
                                       "estimates": beta,
                                       "upper_bound_for_etimates": upper_bounds,
                                       "standard_errors": se_beta})

display(result_table)

plt.plot(lower_bounds)
plt.plot(beta)
plt.plot(upper_bounds)
plt.legend(["lower_bound_for_estimates",
            "estimates",
            "upper_bound_for_estimates"])
plt.show()