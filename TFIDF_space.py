# version: python3.6

from sklearn.datasets.base import Bunch
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def readBunchObj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def writeBunchObj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


def vector_space(bunch_path, space_path, train_tfidf_path=None):
    bunch = readBunchObj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readBunchObj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writeBunchObj(space_path, tfidfspace)
    print("TF-IDF vector space instance was successfully created in,", space_path)


if __name__ == '__main__':
    '''
    bunch_path = "train_word_bag/train_wordbag.dat"
    space_path = "train_word_bag/tfidfspace.dat"
    vector_space(bunch_path, space_path)
    '''

    bunch_path = "test_word_bag2/test_wordbag.dat"
    space_path = "test_word_bag2/testspace.dat"
    train_tfidf_path = "train_word_bag/tfidfspace.dat"
    vector_space(bunch_path, space_path, train_tfidf_path)
