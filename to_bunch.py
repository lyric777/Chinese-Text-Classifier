# version: python3.6

import os
import _pickle as pickle  # 导入序列化
from sklearn.datasets.base import Bunch


def readFile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def readFile_str(path):
    with open(path, "r", encoding="utf8") as fp:
        content = fp.read()
    return content

def toBunch(wordbag_path, seg_path, stopword_path):
    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息
    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    # 获取每个目录下所有的文件
    for subdir in catelist:
        class_path = seg_path + subdir + "/"  # 拼出分类子目录的路径
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        for file_path in file_list:  # 遍历类别目录下文件
            fullname = class_path + file_path  # 拼出文件名全路径
            bunch.label.append(subdir)
            bunch.filenames.append(fullname)
            stpwrdlst = readFile_str(stopword_path).splitlines()  # 去掉停用词再存
            temp = []
            for token in readFile(fullname).split():
                if str(token, 'utf-8') not in stpwrdlst:
                    temp.append(str(token, 'utf-8'))  # 编码一定要变成uft8的
            bunch.contents.append(" ".join(('%s' %i for i in temp)))

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("The construction of the text object ends, in", seg_path)


if __name__ == "__main__":
    '''
    # 对训练集进行Bunch化操作：
    wordbag_path = "train_word_bag/train_wordbag.dat"  # Bunch存储路径
    seg_path = "train_tokens/"  # 分词后分类语料库路径
    stopword_path = "train_word_bag/hlt_stop_words.txt"
    toBunch(wordbag_path, seg_path, stopword_path)
    '''
    stopword_path = "train_word_bag/hlt_stop_words.txt"
    # 对测试集进行Bunch化操作：
    wordbag_path = "test_word_bag2/test_wordbag.dat"  # Bunch存储路径
    seg_path = "test_tokens_even/"  # 分词后分类语料库路径
    toBunch(wordbag_path, seg_path, stopword_path)

