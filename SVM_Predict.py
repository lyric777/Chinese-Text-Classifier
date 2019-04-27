import pickle
from sklearn import svm


def readBunchObj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


trainpath = "train_word_bag/tfidfspace.dat"
train_set = readBunchObj(trainpath)
testpath = "test_word_bag2/testspace.dat"
test_set = readBunchObj(testpath)

from sklearn.externals import joblib

clf = svm.SVC(kernel="poly", class_weight="balanced")  #增加了权重 对比的时候去掉
print("正在计算")
clf.fit(train_set.tdm, train_set.label)

save_dir = "clf_poly.dat"
joblib.dump(clf, save_dir)  # 保存模型
'''
save_dir = "clf_rbf.dat"
joblib.dump(clf, save_dir)  # 保存模型

from sklearn.externals import joblib

clf = joblib.load("F:/Chinese_text_classifier/clf.dat")
'''
print("训练完毕")
predicted = clf.predict(test_set.tdm)
print("预测完毕")

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)

print("SVM预测完毕!!!")

# 计算分类精度：
from sklearn import metrics


def metrics_result(actual, predict):
    print('accuracy:{0:.3f}'.format(metrics.accuracy_score(actual, predict)))
    print('precision:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('recall:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))



metrics_result(test_set.label, predicted)
