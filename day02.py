import nltk
#nltk.download()  使用这个下载需要的语料库
#定义了一个特征提取器，从数据集中抽取特征
def gender_features(word):
    return {'last_letter':word[-1]}
#b=gender_features('shrek')
#print(b)

#拥有了一个特征提取器，需要准备训练的列表和相应的类标签
from nltk.corpus import names
labeled_names=[(name,'male') for name in names.words('male.txt')]+[(name,'female')for name in names.words('female.txt')] #制作数据集
import random
random.shuffle(labeled_names)


#接下来。我们使用特征提取器来处理名称数据，并将特征集列表分为训练集和测试集，训练集用“朴素贝叶斯”分类器
featuresets=[(gender_features(n),gender) for (n,gender) in labeled_names]
train_set,test_set=featuresets[500:],featuresets[:500]
classifier=nltk.NaiveBayesClassifier.train(train_set)#这边训练出这个分类器
b=classifier.classify(gender_features('ali'))
print(b)
print(nltk.classify.accuracy(classifier,test_set))#使用测试集进行测试

#检查分类器，以确定那些特征最有效地区分名称的性别
c=classifier.show_most_informative_features(2)
#这个是似然比
print(c)