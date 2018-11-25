import pandas as pd
import nltk
import string
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords


# 解码方式根据文件的编码方式决定
text = pd.read_csv('spam.csv',encoding='latin-1')
# 提取垃圾短信和正常短信
spam_text = text.loc[text['v1']=='spam']['v2']
ham_text = text.loc[text['v1']=='ham']['v2']

# 建立垃圾短信和正常短信的语料库
# 去掉标点符号
# 将,映射成 空格
trantable = str.maketrans(",",string.punctuation)
def data_clean(text):
    # 对text进行,到空格的映射
    text_clean = text.translate(trantable)
    return text_clean
# 去除stopword
def remove_stopword(text):
    # stopwords.words('english')表示英文停止词表，例如a等不作为索引
    return [word.lower() for word in text if word.lower() not in stopwords.words('english')]
# 给垃圾短信建立语料库并计算词频
spam_corp = []
# 读取数据
for line in spam_text:
    # 去除标点符号
    line_clean = data_clean(str(line))
    # 分词
    word_tk = nltk.word_tokenize(line_clean)
    # 去除停止词
    word_tk_wo_stop = remove_stopword(word_tk)
    # append会将整体加入，extend会将每个元素挑出来加入原list
    spam_corp.extend(word_tk_wo_stop)
