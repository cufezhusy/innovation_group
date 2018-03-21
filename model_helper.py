import pickle
import random
from sklearn.model_selection import train_test_split

def load_test_case():
    with open('test_case.pickle', 'rb') as fp:
        return pickle.load(fp)


def divide_data(X,Y):
    return train_test_split(X, Y,test_size = 0.05, random_state = 42)


if __name__ == '__main__':
    out = load_test_case()
    X = out['X']
    Y = out['Y']
    pos = out['Pos']
    X_train, X_test, y_train, y_test= divide_data(X,Y)

    print(sum(y_test)/len(y_test))