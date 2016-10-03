from sklearn import svm
from utils import load_svm_dataset


def svm_exp(dataset):
    X, y, X_test, y_test = load_svm_dataset(dataset)
    clf = svm.SVC()
    clf.fit(X, y)
    print("Expecting to get:" + y_test)
    predicted = clf.predict(X_test)
    print("Got: " + predicted)
    print("The SVM achieved a score of " + asses_score(y_test, predicted))
    pass


def asses_score(y_test, predicted):
    correct_result = 0
    for i in len(y_test):
        if (y_test[i] == predicted):
            correct_result = correct_result + 1

    return correct_result//len(y_test)
