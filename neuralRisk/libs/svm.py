from sklearn import svm

class SVM(object):
    def __init__(self, dataset):
        """.

        :dataset: the dataset that should be processed
        :returns: The SVM classifier

        """
        classifier = svm.SVC()
        self.dataset = load_dataset_svm(dataset)
        pass
