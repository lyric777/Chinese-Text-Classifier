import os
import jieba
import eel
import _pickle as pickle
from sklearn.datasets.base import Bunch


def readFile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def saveFile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


def mkdir(dir):
    # 去除尾部 \ 符号
    dir = dir.rstrip("\\")  # 不知道要不要去掉
    isExists = os.path.exists(dir)
    if not isExists:
        os.makedirs(dir)
        print('目录创建成功')
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('目录已存在')


def segment(raw_path, seg_path):
    file_list = os.listdir(raw_path)  # 获取未分词语料库中某一类别中的所有文本
    print(file_list)  # 里面有新建的文件夹，问题在这

    for file_path in file_list:  # 遍历类别目录下的所有文件
        if file_path != "tokens":
            fullname = raw_path + file_path  # 拼出文件名全路径如
            content = readFile(fullname)  # 读取文件内容
            content = content.replace("\r\n".encode('utf-8'), "".encode('utf-8'))  # 删除换行
            content = content.replace(" ".encode('utf-8'), "".encode('utf-8'))  # 删除空行、多余的空格
            content_seg = jieba.cut(content)  # 为文件内容分词
            saveFile(seg_path + file_path, " ".join(content_seg).encode('utf-8'))  # 将处理后的文件保存到分词后语料目录



def readFile_str(path):
    with open(path, "r", encoding="utf8") as fp:
        content = fp.read()
    return content


def toBunch(wordbag_path, seg_path, stopword_path):
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], filenames=[], contents=[])
    file_list = os.listdir(seg_path)  # 获取class_path下的所有文件
    for file_path in file_list:  # 遍历目录下文件
        fullname = seg_path + file_path  # 拼出文件名全路径
        bunch.filenames.append(fullname)
        stpwrdlst = readFile_str(stopword_path).splitlines()  # 去掉停用词再存
        temp = []
        for token in readFile(fullname).split():
            if str(token, 'utf-8') not in stpwrdlst:
                temp.append(str(token, 'utf-8'))  # 编码一定要变成uft8的
        bunch.contents.append(" ".join(('%s' % i for i in temp)))

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)


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
    tfidfspace = Bunch(filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readBunchObj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf=True,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writeBunchObj(space_path, tfidfspace)


@eel.expose               # Expose this function to js
def predict(path):
    mkdir(path+"tokens/")
    segment(path, path+"tokens/")  # 不知道中间少不少斜杠
    bunch_path = path + "/wordbag.dat"  # Bunch存储路径
    seg_path = path+"tokens/"  # 分词后分类语料库路径
    stopword_path = "F:/Chinese_text_classifier/train_word_bag/hlt_stop_words.txt"
    toBunch(bunch_path, seg_path, stopword_path)

    space_path = path + "tfidfspace.dat"
    train_tfidf_path = "train_word_bag/tfidfspace.dat"  # 一定要有，比训练多一个，不然维数不一样
    vector_space(bunch_path, space_path, train_tfidf_path)

    test = readBunchObj(space_path)

    from sklearn.externals import joblib
    clf = joblib.load("F:/Chinese_text_classifier/clf.dat")
    predicted = clf.predict(test.tdm)

    for file_name, expct_cate in zip(test.filenames, predicted):
        print(file_name, " -->预测类别:", expct_cate)


predict("F:/Chinese_text_classifier/article/")  # 成功了！！！！
