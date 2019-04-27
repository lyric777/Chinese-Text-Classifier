# version: python3.6


import os
import jieba

# 默认Unicode输出环境


def saveFile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


def readFile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def segment(raw_path, seg_path):
    catelist = os.listdir(raw_path)  # 获取raw_path下的所有子目录

    # 获取每个目录（类别）下所有的文件
    for category in catelist:
        class_path = raw_path + category + "/"  # 拼出分类子目录的路径如：train_corpus/art/
        seg_dir = seg_path + category + "/"  # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/

        if not os.path.exists(seg_dir):  # 是否存在分词目录，如果没有则创建该目录
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本

        for file_path in file_list:  # 遍历类别目录下的所有文件
            fullname = class_path + file_path  # 拼出文件名全路径如：train_corpus/art/21.txt
            content = readFile(fullname)  # 读取文件内容

            content = content.replace("\r\n".encode('utf-8'), "".encode('utf-8'))  # 删除换行
            content = content.replace(" ".encode('utf-8'), "".encode('utf-8'))  # 删除空行、多余的空格
            content_seg = jieba.cut(content)  # 为文件内容分词
            saveFile(seg_dir + file_path, " ".join(content_seg).encode('utf-8'))  # 将处理后的文件保存到分词后语料目录

    print("Chinese corpus word segmentation ends, in", seg_path)


if __name__ == "__main__":
    # 对训练集进行分词
    raw_path = "./train_set/"  # 未分词分类语料库路径
    seg_path = "./train_tokens/"  # 分词后分类语料库路径
    segment(raw_path, seg_path)

    # 对测试集进行分词
    raw_path = "./test_set/"  # 未分词分类语料库路径
    seg_path = "./test_tokens/"  # 分词后分类语料库路径
    segment(raw_path, seg_path)
