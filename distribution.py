from collections import Counter
from sklearn.datasets.base import Bunch
import _pickle as pickle


def readBunchObj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


#bunch = readBunchObj("train_word_bag/train_wordbag.dat")
bunch = readBunchObj("test_word_bag/test_wordbag.dat")
sample = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames)


print(Counter(sample.label))
