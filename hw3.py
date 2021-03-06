import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pystan


os.getcwd()
os.chdir("KocPython2019/Homework/")

#open and manipulate the data
data = pd.read_csv("trend2.csv") #open file 
data.columns = data.columns.map(str.strip)
data = data.dropna() #list-wise deletion

#lookup table (dict) for each unique county, for indexing.
data.country = data.country.map(str.strip)
countries = data.country.unique()
n = len(countries)
country_lookup = dict(zip(countries, range(n)))

#create local copies of variables
country = data['country_code'] = data.country.replace(country_lookup).values
religiosity = data.church2
inequality = data.gini_net.values
rgdpl = data.rgdpl.values


#varying intercept model
varying_intercept = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> country[N]; 
  vector[N] x1; //inequality
  vector[N] x2; //rgdpl
  vector[N] y; //religiosity
}
parameters {
  vector[J] a;
  real b1;
  real b2;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
}
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = a[country[i]] + x1[i] * b1 + x2[i] * b2;
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal(mu_a, sigma_a);
  b1 ~ normal(0,1);
  b2 ~ normal(0,1);
  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""

varying_intercept_data = {'N': len(religiosity),
                          'J': len(countries),
                          'country': country+1,
                          'x1': inequality,
                          'x2': rgdpl,
                          'y': religiosity}

varying_intercept_fit = pystan.stan(model_code=varying_intercept, data=varying_intercept_data, iter=1000, chains=2)

a_sample = pd.DataFrame(varying_intercept_fit['a'])
