# Examples stolen from: 
# http://scikit-learn.org/stable/modules/ensemble.html
# http://scikit-learn.org/stable/auto_examples/exercises/plot_digits_classification_exercise.html

from sklearn import datasets, neighbors, linear_model, svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

n_samples = len(X_digits)

X_train = X_digits[:.9 * n_samples]
y_train = y_digits[:.9 * n_samples]
X_test = X_digits[.9 * n_samples:]
y_test = y_digits[.9 * n_samples:]

knn = neighbors.KNeighborsClassifier(weights='distance')
logistic = linear_model.LogisticRegression()
dt = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
ab = AdaBoostClassifier(dt, n_estimators=300)

rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
et = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
support_vector_machine = svm.SVC(gamma=0.001)


print('KNN: %f ' %cross_val_score(knn, X_train, y_train, n_jobs=-1).mean())
print('Logistic: %f ' %cross_val_score(logistic, X_train, y_train, n_jobs=-1).mean())
print('Decision Tree: %f ' %cross_val_score(dt, X_train, y_train, n_jobs=-1).mean())
print('Random Forest: %f ' %cross_val_score(rf, X_train, y_train, n_jobs=-1).mean())
print('Extra trees: %f ' %cross_val_score(et, X_train, y_train, n_jobs=-1).mean())
print('Ada Boost: %f ' %cross_val_score(ab, X_train, y_train, n_jobs=-1).mean())
print('SVM: %f ' %cross_val_score(support_vector_machine, X_train, y_train, n_jobs=-1).mean())