import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

def replace_non_numeric(df):
	df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
	df["Embarked"] = df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)	
	return df

train_df = replace_non_numeric(pd.read_csv("train.csv"))

et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

columns = ["Fare", "Pclass", "Sex"]

labels = train_df["Survived"].values
features = train_df[list(columns)].values
	
et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()

print("{0} -> ET: {1})".format(columns, et_score))

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(features)

test_df = replace_non_numeric(pd.read_csv("test.csv"))

et.fit(features, labels)

predictions = et.predict(imp.transform(test_df[columns].values))
test_df["Survived"] = pd.Series(predictions)
test_df.to_csv("foo.csv", cols=['PassengerId', 'Survived'], index=False)