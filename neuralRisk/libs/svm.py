from sklearn import svm
from utils import load_svm_dataset


def svm_exp(dataset):
    datasets = load_svm_dataset(dataset)
    X, y, X_test, y_test = datasets
    clf = svm.SVC()
    clf.fit(X, y)
    print("Expecting to get:" + str(y_test))
    predicted = clf.predict(X_test)
    print("Got: " + str(predicted))
    print("The SVM achieved a score of " + str(asses_score(y_test, predicted))
          + "%")


def asses_score(y_test, predicted):
    correct_result = 0
    print(type(y_test))
    for i in range(y_test.shape[0]):
        if (y_test[i] == predicted[i]):
            correct_result = correct_result + 1
    return (correct_result/(y_test.shape[0]))*100
