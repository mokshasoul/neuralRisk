import csv
from sklearn.feature_extraction import DictVectorizer


def main():
    with open('./bank-full.csv', 'rb') as csvinput:
        csv_dict = csv.DictReader(csvinput)
    return


main()
