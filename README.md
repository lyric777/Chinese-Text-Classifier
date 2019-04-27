# Chinese-Text-Classifier
基于SVM中文文本分类

复旦大学中文语料库，使用15类

语料文件太大，可以在这里下载http://www.nlpir.org/wordpress/

训练时间太长，所以保存了一份模型（总共训练了3个模型，线性核的分类效果最好）

最后使用的接口还没做好，大致就是那个意思了，没时间改了

分两个部分，训练和测试的一步步运行：get_tokens --> to_bunch --> TFIDF_space --> SVM_Predict

也可以直接使用训练好的模型，直接运行use.py，建文件夹F:/Chinese_text_classifier/article/，最后在article文件夹里放要分类的文章(.txt或者.dat之类的)
