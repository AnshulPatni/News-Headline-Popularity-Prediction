# News-Headline-Popularity-Prediction
---

## 1. Introduction

News headlines belonging to certain categories can trend on the social media platforms like Facebook, Google+ and LinkedIn based on the sentiment and the time of these news. The trends have been constantly changing and this could predict the future trends amongst the users of these platforms.

To understand the trends, we will start with the news articles that became popular in the past to find the underlying patterns, and apply the findings on the
upcoming news to predict the popularity of that news article. We will also find the sentiment of the news article and how does that impact the trends on the social media platforms.


### 1.1 Motivation

In order to capture the reader’s attention to influence the online experience of reading news, headlines play a
vital role. The studies have confirmed this behaviour empirically where many users attends the news
headlines to ascertain the overview of any article, but then they exhibit no reading activities. Furthermore,
there are many online spaces where headlines are the only visible part of the news article; for example news
feeds and social media. Yet despite this, headlines have not been considered before as the sole source of
data for news article popularity prediction. Most models make use of post-publication data, such as the
number of early adopters.


### 1.2 Objective

Our goal is to investigate a wide variety of text features
extracted from headlines and determine whether they have impact on social media popularity of news articles.
We enhance prior work by: (i) using only headlines; (ii) introducing new features; and (iii) using a
source-internal evaluation.


### 1.3 Literature/Market Review

The dataset that we have used here has been used from a publication with below details:<br>
a. UCI link to dataset -
http://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms<br>
b. Directory Structure of Dataset - http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/<br>
c. This dataset contains the details of the news headlines in different categories with their sentiments and the
popularity quotient of these news on different social media platforms spanning November 2015 - July 2016 <br>
d. The dataset for news details includes news article ID, title, headline, source, topic, publish date, sentiment
score for title, sentiment score for headline.<br>
e. The dataset for multiple social media platforms are categorized according to the news categories and the
details includes new article ID, popularity in time slices 0 min - 2 days upon publication with difference of
20 min.



## 2. System Design & Implementation details

### 2.1 Algorithm(s) considered/selected

There are various classification, regression & time-series predictive algorithms to solve this problem.

#### Algorithms used :

a. **Linear SVC** - The objective of clustering is to partition a data set into groups according to some
criterion in an attempt to organize data into a more meaningful form. There are many ways of
achieving this goal. Clustering may proceed according to some parametric model or by grouping points
according to some distance or similarity measure as in hierarchical clustering . A natural way to put
cluster boundaries is in regions in data space where there is little data, i.e. in "valleys" in the probability
distribution of the data. This is the path taken in support vector clustering (SVC), which is based on
the support vector approach.
In SVC data points are mapped from data space to a high dimensional feature space using a
kernel function. In the kernel feature space the algorithm searches for the smallest sphere that
encloses the image of the data using the Support Vector Domain Description algorithm. This sphere,
when mapped back to data space, forms a set of contours which enclose the data points. Those
contours are then interpreted as cluster boundaries, and points enclosed by each contour are
associated by SVC to the same cluster.

b. **Multinomial Naive Bayes Classifier** - These classifiers are a family of simple " probabilistic classifiers
"based on applying Bayes' theorem with strong (naive) independence assumptions between the
features.It is a popular method for text categorization , the problem of judging documents as belonging
to one category or the other (such as spam or legitimate , sports or politics, etc.) with word frequencies
as the features. With appropriate pre-processing, it is competitive in this domain with more advanced
methods including support vector machines . It also finds application in automatic medical diagnosis .
Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of
variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by
evaluating a closed-form expression , which takes linear time , rather than by expensive iterative
approximation as used for many other types of classifiers.

c. **Random Forest** - Random forests or random decision forests are an ensemble learning method for
classification , regression and other tasks, that operate by constructing a multitude of decision trees at
training time and outputting the class that is the mode of the classes (classification) or mean prediction
(regression) of the individual trees. ] Random decision forests correct for decision trees' habit of
overfitting to their training set.

d. **NLP using Wordnet**[1]- It’s common in the world on Natural Language Processing to need to
compute sentence similarity. Wordnet is a great tool. NLTK includes the English WordNet, with
155,287 words and 117,659 synonym sets. I’ll begin by looking at synonyms and how they are
accessed in WordNet. Some of the properties that make Wordnet so useful: Synonyms are grouped
together in something called Synset. There are hierarchical links between synsets (ISA relations or
hypernym/hyponym relations)

e. **Sequence Matcher** - his is a flexible class for comparing pairs of sequences of any type, so long as
the sequence elements are hashable . The idea is to find the longest contiguous matching
subsequence that contains no junk elements. The same idea is then applied recursively to the pieces
of the sequences to the left and to the right of the matching subsequence. This does not yield minimal
edit sequences, but does tend to yield matches that look right to people.

f. **Cosine Similarity** - Cosine similarity is a measure of similarity between two non-zero vectors of an
inner product space that measures the cosine of the angle between them. The cosine of 0° is 1, and it
is less than 1 for any other angle in the interval [0,2π). It is thus a judgment of orientation and not
magnitude: two vectors with the same orientation have a cosine similarity of 1, two vectors at 90° have
a similarity of 0, and two vectors diametrically opposed have a similarity of -1, independent of their
magnitude. Cosine similarity is particularly used in positive space, where the outcome is neatly
bounded in [0,1]. The name derives from the term "direction cosine": in this case, note that unit vectors
are maximally "similar" if they're parallel and maximally "dissimilar" if they're orthogonal
(perpendicular).

g. **ARIMA** - ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. It is a class
of model that captures a suite of different standard temporal structures in time series data. An ARIMA
model is a class of statistical models for analyzing and forecasting time series data. It explicitly caters
to a suite of standard structures in time series data, and as such provides a simple yet powerful
method for making skillful time series forecasts.

h. **Prophet Forecast**[4] - Prophet is a procedure for forecasting time series data. It is based on an
additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It
works best with daily periodicity data with at least one year of historical data. Prophet is robust to
missing data, shifts in the trend, and large outliers.

i. **LSTM**[2] - The Long Short-Term Memory network or LSTM network is a type of recurrent neural
network used in deep learning because very large architectures can be successfully trained.In this I
have developed a number of LSTMs for a standard time series prediction problem.

j. **Moving average (MA) models**[3] - A moving average is a technique to get an overall idea of the trends
in a data set; it is an average of any subset of numbers. The moving average is extremely useful for
forecasting long-term trends. You can calculate it for any period of time. A moving average is a
technique to get an overall idea of the trends in a data set; it is an average of any subset of numbers.
The moving average is extremely useful for forecasting long-term trends. You can calculate it for any
period of time.

### 2.2 Technologies & Tools used

**Programming language**: Python 3.6
**Development tool**: Jupyter notebook
**Libraries**: Numpy(v..14.3), Scipy(v..19.1), Pandas(v. 0.20.3), nltk(v. 3.2.4), sklearn(v. 0.19.1), matplotlib(v.2.1.0) Tensorflow(v. 1.5.1), Keras(v. 2.1.5)

### 2.2 Architecture design

![alt text](https://github.com/abhishek-yadav-cse/news-headline-popularity-prediction/blob/master/architecture.png)


### 2.3 System design/architecture/data flow

We have designed the model using the following steps-

a. Identify Dataset

b. Data extraction

c. Preprocessing

d. Classification

e. Finding nearest similar data points

f. Forecast data


## 3. Experiments/ Proof of concept evaluation

### 3.1 Dataset Used

a. **_UCI link to dataset_** -
http://archive.ics.uci.edu/ml/datasets/News+Popularity+in+Multiple+Social+Media+Platforms

b. **_Directory Structure of Dataset_** - http://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/

c. This dataset contains the details of the news headlines in different categories with their sentiments and the popularity quotient of these news on different social media platforms spanning November 2015 - July 2016

d. The dataset for news details includes news article ID, title, headline, source, topic, publish date, sentiment score for title, sentiment score for headline.

e. The dataset for multiple social media platforms are categorized according to the news categories and the details includes new article ID, popularity in time slices 0 min - 2 days upon publication with difference of 20 min.

### 3.2 Methodology followed

1. Read news data and and any social network data(say facebook). Social network data has timestamps over a period.

2. For each record merge data from facebook dataset and news dataset on ID.

3. Use Countvectorizer to tokenize the documents and count the occurrences of token and
return them as a sparse matrix

4. Use different classifiers predict the category of news headline and then take the classifier with highest F1 score.

5. Next step is to find similarity matching of test sentence with different other sentences in the same category.

6. Use LSTM, Moving Average, Prophet, ARIMA model to find pattern in present data and predict future.

7. Result of different time series analysis is shown and compared

### 3.3 Analysis of results

Comparative study of different classification algorithms that I used:

![alt text](https://github.com/abhishek-yadav-cse/news-headline-popularity-prediction/blob/master/comparisonTable.png)


### 3.4 Graphs for Algorithm Comparison

![alt text](https://github.com/abhishek-yadav-cse/news-headline-popularity-prediction/blob/master/graphSet1.png)
![alt text](https://github.com/abhishek-yadav-cse/news-headline-popularity-prediction/blob/master/graphSet2.png)
![alt text](https://github.com/abhishek-yadav-cse/news-headline-popularity-prediction/blob/master/graphSet3.png)



## 4. Results

### 4.1 Things that worked

+ Classification model(Decision Tree, Naive Bayes, Linear SVC etc.) worked perfectly fine with the highest F1 score of 0.95.

+ Similarity Matching models (wordnet, sequence matcher, cosine similarity) for sentences worked quite well.

+ Time Series Analysis models like LSTM, ARIMA, MA, Prophet model worked well and showed good
result.

### 4.2 Things that didn’t work well

+ Combined model of ARIMA + MA didn’t worked.

+ k-Neighbours classifier didn’t worked well may be because of the size of data.

+ Reducing the curse of dimensionality was not accomplished as the result is showing NAN values.



## References:
[1]. [Wordnet](https://nlpforhackers.io/starting-wordnet/)
[2]. [LSTM Time Series](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
[3]. [Moving Average in Time Series Analysis](https://ieeexplore.ieee.org/document/6708545/)
[4]. [Prophet](https://facebook.github.io/prophet/docs/quick_start.html)
[5]. [Machine Learning with Text - Count Vectorizer Sklearn : Youtube Link](https://www.youtube.com/watch?v=RZYjsw6P4nI)
[6]. [KNN Implementation](http://dataaspirant.com/2016/12/30/k-nearest-neighbor-implementation-scikit-learn/)
[7]. [KNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
[8]. [Wordnet Sentence Similarity](https://nlpforhackers.io/wordnet-sentence-similarity/)
[9]. [Time Series Forecasting](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
[10]. [Time Series Forecasting with Prophet](https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-prophet-in-python-3)
[12]. [Time Series Analysis](https://www.kaggle.com/niyamatalmass/machine-learning-for-time-series-analysis)