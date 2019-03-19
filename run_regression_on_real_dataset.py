from linear_regression import linear_regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("winequality-white.csv", sep=";")

covariates = df.drop("quality", axis=1).values
targets = df["quality"].values

beta, se_beta, lower_bounds, upper_bounds = linear_regression(covariates, targets)

result_table = pd.DataFrame.from_dict({"lower_bound_for_estimates": lower_bounds,
                                       "estimates": beta,
                                       "upper_bound_for_etimates": upper_bounds,
                                       "standard_errors": se_beta})

print("Result table:")
display(result_table)

plt.plot(lower_bounds)
plt.plot(beta)
plt.plot(upper_bounds)
plt.title("Result plot")
plt.legend(["lower_bound_for_estimates",
            "estimates",
            "upper_bound_for_estimates"])