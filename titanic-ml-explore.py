import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm, neighbors
import itertools as it

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def all_combinations(coll):
	return reduce(lambda acc, x: acc + list(it.combinations(coll, x)), range(1, len(coll) + 1), [])

def right_justify(s):
    return "%45s" % str(s)

train_df = pd.read_csv("train.csv")
train_df["Sex"] = train_df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
train_df["Embarked"] = train_df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)

meanAge=np.mean(train_df.Age)
train_df.Age=train_df.Age.fillna(meanAge)
train_df.Cabin = train_df.Cabin.fillna('Unknown')

n_samples = len(train_df)

rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
knn = neighbors.KNeighborsClassifier(weights='distance')
support_vector_machine = svm.SVC(gamma=0.001)
et = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)

print(train_df.head())

all_columns = ["Fare", "Sex", "Pclass", 'Embarked', "Age"]


for columns in all_combinations(all_columns):
	labels = train_df["Survived"].values
	features = train_df[list(columns)].values

	labels_train = labels[:.9 * n_samples]
	labels_test = labels[.9 * n_samples:]

	features_train = features[:.9 * n_samples]
	features_test = features[.9 * n_samples:]
	
	rf_score = cross_val_score(rf, features_train, labels_train, n_jobs=-1).mean()
	et_score = cross_val_score(et, features_train, labels_train, n_jobs=-1).mean()
	svm_score = cross_val_score(support_vector_machine, features_train, labels_train, n_jobs=-1).mean()
	knn_score = cross_val_score(knn, features_train, labels_train, n_jobs=-1).mean()

	columns = right_justify(columns)

	print("{0} -> {5} (RF: {1}, ET: {2}, SVM: {3}, KNN: {4})".format(columns, rf_score, et_score, svm_score, knn_score, max([rf_score, et_score, svm_score, knn_score])))	