from sklearn. linear_model import Perceptron 
# Import a routine for fetching the 20 newsgroups dataset from sklearn.
from sklearn. datasets import fetch_20newsgroups

# Limit the categories of the dataset.
categories = ['alt.atheism', 'sci.med']


# Obtain documents for our category selection
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

# Our perceptron is defined. It will be trained for 100 iterations.
perceptron = Perceptron (max_iter=100)

#The familiar CountVectorizer is fit on our training data
from sklearn. feature_extraction. text import CountVectorizer
cv = CountVectorizer ()
X_train_counts = cv. fit_transform(train.data)

# Load, fit, and deploy a TF. IDF transformer from sklearn. It computes TF.IDF representations of TF.IDF vectors.
from sklearn. feature_extraction.text import TfidfTransformer
tfidf_tf = TfidfTransformer()
X_train_tfidf = tfidf_tf.fit_transform(X_train_counts)


# The perceptron is trained on the TF.IDF vector.
perceptron. fit (X_train_tfidf, train. target) 

# Our test Data
test_docs = ['Religion is widespread, even in modern times','His kidneyà failed', 'The pope is a controversial leader', 'White blood cells fightaoff infections', 'The reverend had a heart attack in church']


# The test data is vectorized first to count vectors and then to TF.IDF vectors.
X_test_counts = cv. transform (test_docs)
X_test_tfidf = tfidf_tf. transform (X_test_counts)


# The perceptron is applied to the test documents
pred = perceptron.predict (X_test_tfidf)

# The results are printed
for doc, category in zip (test_docs, pred) :
    print(' %r => %s' % (doc, train. target_names [category]))

