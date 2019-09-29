import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from nyoka import skl_to_pmml
import random

data = []

ITERATIONS = 200

# Approved
lenovoPrices = np.random.normal(loc=1500, scale=40, size=ITERATIONS)
for price in lenovoPrices:
    data.append({'ActorId': 'John', 'item': 'Lenovo', 'price': price, 'approved': 1})

lenovoPrices = np.random.normal(loc=1500, scale=40, size=ITERATIONS)
for price in lenovoPrices:
    data.append({'ActorId': 'Mary', 'item': 'Lenovo', 'price': price, 'approved': 1})

applePrices = np.random.normal(loc=2500, scale=40, size=ITERATIONS)
for price in applePrices:
    data.append({'ActorId': 'John', 'item': 'Apple', 'price': price, 'approved': 1})

applePrices = np.random.normal(loc=2500, scale=40, size=ITERATIONS)
for price in applePrices:
    data.append({'ActorId': 'Mary', 'item': 'Apple', 'price': price, 'approved': 1})

# Rejected
applePrices = np.random.normal(loc=2500, scale=40, size=ITERATIONS)
for price in applePrices:
    data.append({'ActorId': 'John', 'item': 'Lenovo', 'price': price, 'approved': 0})

applePrices = np.random.normal(loc=2500, scale=40, size=ITERATIONS)
for price in applePrices:
    data.append({'ActorId': 'Mary', 'item': 'Lenovo', 'price': price, 'approved': 0})

df = pd.DataFrame(data)
df.ActorId = pd.Categorical(pd.factorize(df.ActorId)[0])
df.approved = pd.Categorical(pd.factorize(df.approved)[0])
df.item = pd.Categorical(pd.factorize(df.item)[0])


# Split data
outputs = df['approved']
inputs = df[['ActorId', 'price', 'item']]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4,
                                                    random_state=23)

# Random forest model
rf = RandomForestRegressor()
pipeline = Pipeline([("regressor", rf)])
pipeline.fit(X_train, y_train)

# save PMML model
skl_to_pmml(pipeline, ['ActorId', 'price', 'item'], 'approved', "models/random_forest.pmml")

# Logistic regression model
lr = LogisticRegression()
pipeline = Pipeline([("regressor", lr)])
pipeline.fit(X_train, y_train)

# save PMML model
skl_to_pmml(pipeline, ['ActorId', 'price', 'item'], 'approved', "models/logistic_regression.pmml")