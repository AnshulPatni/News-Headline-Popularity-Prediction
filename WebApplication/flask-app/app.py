from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import scipy
import sklearn
import pandas as pd
import nltk
import csv
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC , NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from operator import is_not
from functools import partial
import collections
import operator
import itertools
import csv
import pandas as pd
import matplotlib.pyplot as plot
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from flask import send_file
import tempfile
import pygal
import random
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.dates as mdates
import seaborn as sns
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


popularity_average =[]

vocabulary = ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

app = Flask(__name__)
value_category = []
entered_sentence = ['']


@app.route('/')
def home():
	return render_template('home.html')

traindata = pd.read_csv("final_news.csv")
train_facebook_economy = pd.read_csv("dataset/Facebook_Economy.csv")
train_facebook_microsoft = pd.read_csv("dataset/Facebook_Microsoft.csv")
train_Facebook_Obama = pd.read_csv("dataset/Facebook_Obama.csv")
train_Facebook_Palestine = pd.read_csv("dataset/Facebook_Palestine.csv")

bigdata_facebook = pd.concat([train_facebook_economy, train_facebook_microsoft, train_Facebook_Obama, train_Facebook_Palestine], ignore_index=True)
facebook_merge = pd.merge(traindata, bigdata_facebook, on="IDLink")

X = facebook_merge.Headline
y = facebook_merge.Topic
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

vector = CountVectorizer()
count = vector.fit_transform(x_train.values.astype('U'))

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(count)

txt_classifier = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), analyzer='char')),
                    ('tfidf', TfidfTransformer()),
                    ('dtc', LinearSVC()),
])
txt_classifier = txt_classifier.fit(x_train.values.astype('U'), y_train.values.astype('U'))


@app.route('/getdelay',methods=['POST','GET'])
def get_delay():

    if request.method=='POST':
        result=request.form['headline']
        data = [result]
        entered_sentence = data
        result1 = txt_classifier.predict(data)
        value_category = result1
        result_sentences = similarity(value_category)
        popularity_average = popularity(result_sentences)
        timelist = []
        for i in range (1,148):
            timelist.append(i)
        graph_original = pygal.Line()
        graph_original.title = 'Original time analysis graph'
        graph_original.add('Original Prediction',  popularity_average)
        graph_original_plot = graph_original.render_data_uri()
        return render_template('result.html',prediction=result1, entered_sentence=entered_sentence, result_sentences=result_sentences,graph_data1 = graph_original_plot, name = result.upper())

def similarity(value_category):
    face_head = facebook_merge.Headline[facebook_merge.Topic==value_category[0]]
    face_head.isnull().sum()
    face_headline = face_head.dropna()
    names,cls = [],[]
    for n in face_headline:
        names.append(n)
    for n in facebook_merge.Topic:
        cls.append(n)
    names = names[:70000]
    i = 0
    a,sorted_list = {},[]
    for sentence in names:
        i = i+1
        a[sentence] = sentence_similarity(entered_sentence, sentence)
    d = collections.Counter(a)
    f = d.most_common(5)
    res_list = [x[0] for x in f]
    sorted_list = sorted(a.items(), key=operator.itemgetter(1), reverse = True)
    sorted_list1 = [i[0] for i in sorted_list]
    list3 = sorted_list1[:5]
    return list3
 
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    sentence3 = []
    sentence2 = sentence2.split(' ')
    for i in range(len(sentence2)):
        if sentence2[i].isalpha():
            sentence3.append(sentence2[i])
    str1 = ""
    for i in sentence3:
        str1 += i + ','
    sentence1 = pos_tag(word_tokenize(str(sentence1)))
    str1 = pos_tag(word_tokenize(str1))
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in str1]
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    for synset in synsets1:
            simlist = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss) is not None]
            if not simlist:
                continue
            best_score = max(simlist)

            score += best_score
            count += 1
    if count!=0:
        score /= count
    return score

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None


def popularity(list3):
    list_tsp_final,list_final = [],[]
    for i in range(len(list3)):
        list_tsp = []
        a = facebook_merge[facebook_merge.Headline == list3[i]]
        b = a.values.T.tolist()
        list_tsp.append(b[8:])
        list_tsp_final.append(list_tsp)
    list3_one = list_tsp_final[0][0]
    list3_two = list_tsp_final[1][0]
    list3_three = list_tsp_final[2][0]
    list3_four = list_tsp_final[3][0]
    list3_five = list_tsp_final[4][0]
    list4_one =  list(itertools.chain.from_iterable(list3_one))
    list4_two =  list(itertools.chain.from_iterable(list3_two))
    list4_three =  list(itertools.chain.from_iterable(list3_three))
    list4_four = list(itertools.chain.from_iterable(list3_four))
    list4_five = list(itertools.chain.from_iterable(list3_five))
    for i in range(len(list3_one)):
        list_final.append((list4_one[i]+list4_two[i] + list4_three[i] + list4_four[i] + list4_five[i])/5)
    return list_final

@app.route('/graph',methods=['GET'])
# ...do stuff...
def myplot():
    time = [0, 1, 2, 3, 4, 5]
    popularity = [0, 10, 20, 30, 40, 50]

    plt.plot(time, popularity)
    plt.xlabel('Time (min)')
    plt.ylabel('Popularity (self-defined unit)')
    
    f = tempfile.TemporaryFile()
    plt.savefig(f)    
    img64 = base64.b64encode(f.read()).decode('UTF-8')
    f.close()
    return render_template('graph.html', image_data=img64)


@app.route('/pygalexample/')
def pygalexample():	
    print(popularity_average)
    graph1 = pygal.Line()
    graph1.title = 'Original time analysis graph'
    graph1.x_labels = ['2011-01-01','2011-01-02','2011-01-03','2011-01-04','2011-01-05','2011-01-06','2011-01-07','2011-01-08','2011-01-09','2011-01-10','2011-01-11','2011-01-12','2011-01-13','2011-01-14','2011-01-15','2011-01-16','2011-01-17','2011-01-18','2011-01-19','2011-01-20','2011-01-21','2011-01-22','2011-01-23','2011-01-24','2011-01-25','2011-01-26','2011-01-27','2011-01-28','2011-01-29','2011-01-30','2011-01-31','2011-02-01','2011-02-02','2011-02-03','2011-02-04','2011-02-05','2011-02-06','2011-02-07','2011-02-08','2011-02-09','2011-02-10','2011-02-11','2011-02-12','2011-02-13','2011-02-14','2011-02-15','2011-02-16','2011-02-17','2011-02-18','2011-02-19','2011-02-20','2011-02-21','2011-02-22','2011-02-23','2011-02-24','2011-02-25','2011-02-26','2011-02-27','2011-02-28','2011-03-01','2011-03-02','2011-03-03','2011-03-04','2011-03-05','2011-03-06','2011-03-07','2011-03-08','2011-03-09','2011-03-10','2011-03-11','2011-03-12','2011-03-13','2011-03-14','2011-03-15','2011-03-16','2011-03-17','2011-03-18','2011-03-19','2011-03-20','2011-03-21','2011-03-22','2011-03-23','2011-03-24','2011-03-25','2011-03-26','2011-03-27','2011-03-28','2011-03-29','2011-03-30','2011-03-31','2011-04-01','2011-04-02','2011-04-03','2011-04-04','2011-04-05','2011-04-06','2011-04-07','2011-04-08','2011-04-09','2011-04-10','2011-04-11','2011-04-12','2011-04-13','2011-04-14','2011-04-15','2011-04-16','2011-04-17','2011-04-18','2011-04-19','2011-04-20','2011-04-21','2011-04-22','2011-04-23','2011-04-24','2011-04-25','2011-04-26','2011-04-27','2011-04-28','2011-04-29','2011-04-30','2011-05-01','2011-05-02','2011-05-03','2011-05-04','2011-05-05','2011-05-06','2011-05-07','2011-05-08','2011-05-09','2011-05-10','2011-05-11','2011-05-12','2011-05-13','2011-05-14','2011-05-15','2011-05-16','2011-05-17','2011-05-18','2011-05-19','2011-05-20','2011-05-21','2011-05-22','2011-05-23','2011-05-24','2011-05-25','2011-05-26','2011-05-27']
    graph1.add('Python',  popularity_average)
    graph_data1 = graph1.render_data_uri()
    new_plot_value = myplot(popularity_average)
    graph2 = pygal.Line()
    graph2.title = 'Original time analysis graph'
    graph2.x_labels = ['2011-01-01','2011-01-02','2011-01-03','2011-01-04','2011-01-05','2011-01-06','2011-01-07','2011-01-08','2011-01-09','2011-01-10','2011-01-11','2011-01-12','2011-01-13','2011-01-14','2011-01-15','2011-01-16','2011-01-17','2011-01-18','2011-01-19','2011-01-20','2011-01-21','2011-01-22','2011-01-23','2011-01-24','2011-01-25','2011-01-26','2011-01-27','2011-01-28','2011-01-29','2011-01-30','2011-01-31','2011-02-01','2011-02-02','2011-02-03','2011-02-04','2011-02-05','2011-02-06','2011-02-07','2011-02-08','2011-02-09','2011-02-10','2011-02-11','2011-02-12','2011-02-13','2011-02-14','2011-02-15','2011-02-16','2011-02-17','2011-02-18','2011-02-19','2011-02-20','2011-02-21','2011-02-22','2011-02-23','2011-02-24','2011-02-25','2011-02-26','2011-02-27','2011-02-28','2011-03-01','2011-03-02','2011-03-03','2011-03-04','2011-03-05','2011-03-06','2011-03-07','2011-03-08','2011-03-09','2011-03-10','2011-03-11','2011-03-12','2011-03-13','2011-03-14','2011-03-15','2011-03-16','2011-03-17','2011-03-18','2011-03-19','2011-03-20','2011-03-21','2011-03-22','2011-03-23','2011-03-24','2011-03-25','2011-03-26','2011-03-27','2011-03-28','2011-03-29','2011-03-30','2011-03-31','2011-04-01','2011-04-02','2011-04-03','2011-04-04','2011-04-05','2011-04-06','2011-04-07','2011-04-08','2011-04-09','2011-04-10','2011-04-11','2011-04-12','2011-04-13','2011-04-14','2011-04-15','2011-04-16','2011-04-17','2011-04-18','2011-04-19','2011-04-20','2011-04-21','2011-04-22','2011-04-23','2011-04-24','2011-04-25','2011-04-26','2011-04-27','2011-04-28','2011-04-29','2011-04-30','2011-05-01','2011-05-02','2011-05-03','2011-05-04','2011-05-05','2011-05-06','2011-05-07','2011-05-08','2011-05-09','2011-05-10','2011-05-11','2011-05-12','2011-05-13','2011-05-14','2011-05-15','2011-05-16','2011-05-17','2011-05-18','2011-05-19','2011-05-20','2011-05-21','2011-05-22','2011-05-23','2011-05-24','2011-05-25','2011-05-26','2011-05-27']
    graph2.add('Python',  new_plot_value)
    graph_data2 = graph2.render_data_uri()
    return render_template("graph.html", graph_data1 = graph_data1, graph_data2 = graph_data2)

if __name__ == '__main__':
	app.debug = True
	app.run()