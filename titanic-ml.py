import pandas as pd
import numpy as np
import itertools as it

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm, neighbors
from sklearn.preprocessing import Imputer

def replace_non_numeric(df):
	df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
	df["Embarked"] = df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)	
	return df

train_df = replace_non_numeric(pd.read_csv("train.csv"))

n_samples = len(train_df)

et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

all_columns = ["Fare", "Sex", "Pclass", 'Embarked']

labels = train_df["Survived"].values
features = train_df[all_columns].values

et.fit(features, labels)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)

test_df = replace_non_numeric(pd.read_csv("test.csv"))
test_features = imp.transform(test_df[all_columns].values)

test_df["Survived"] = pd.Series(et.predict(test_features))
test_df.to_csv("result-ml.csv", cols=['PassengerId', 'Survived'], index=False)