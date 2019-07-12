import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from nyoka import skl_to_pmml
import random

data = []

for i in range(10000):
    r = random.random()
    if r < 0.9:
        data.append({'ActorId': 'john', 'level': 5, 'approved': 0})
    else:
        data.append({'ActorId': 'john', 'level': 5, 'approved': 1})

    v = random.random()
    if r < 0.9:
        data.append({'ActorId': 'mary', 'level': 5, 'approved': 1})
    else:
        data.append({'ActorId': 'mary', 'level': 5, 'approved': 0})

df = pd.DataFrame(data)
df.ActorId = pd.Categorical(pd.factorize(df.ActorId)[0])
df.level = pd.Categorical(pd.factorize(df.level)[0])
df.approved = pd.Categorical(pd.factorize(df.approved)[0])

# Split data
outputs = df['approved']
inputs = df[['ActorId', 'level']]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.4,
                                                    random_state=23)

# Random forest model
rf = RandomForestRegressor()
pipeline = Pipeline([("regressor", rf)])
pipeline.fit(X_train, y_train)

# save PMML model
skl_to_pmml(pipeline, ['ActorId', 'level'], 'approved', "models/random_forest.pmml")

# Logistic regression model
lr = LogisticRegression()
pipeline = Pipeline([("regressor", lr)])
pipeline.fit(X_train, y_train)

# save PMML model
skl_to_pmml(pipeline, ['ActorId', 'level'], 'approved', "models/logistic_regression.pmml")
