from flask import Flask, render_template, request
import pickle
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
import warnings
import nltk
warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def text_preprocessing(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stopwords_en = stopwords.words('english')
    stemmerSnowball = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    text = re.sub("@[A-Za-z0-9_]+", "", text)
    text = re.sub("#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"\<.*\/\>", '', text)
    text = text.encode("ascii", "ignore")
    text = text.decode()
    text = re.sub(r'\w*\d\w*', "", text)
    text = ' '.join([i for i in text.split() if len(i) > 2])
    remove_html_tags = [re.sub(r"\<.*\>", '', word)
                        for word in text.split(" ")]
    remove_bracket_words = [re.sub(r"\(.*\)", '', word)
                            for word in remove_html_tags]
    text = contractions.fix((str(" ".join(remove_bracket_words)).lower()))
    words = [re.sub(r'[^\w\s]', '', word) for word in text.split(' ')]
    without_stop_words = [re.sub(r'[^\w+]', '', word)
                          for word in words if word not in stopwords_en]
    stemmedWords = [stemmerSnowball.stem(word) for word in without_stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmedWords]
    tokenized = [" ".join(tokenizer.tokenize(word)) for word in lemmatized]

    return re.sub(' +', ' ', str(" ".join(tokenized)))


def featureVecMethod(words, model, num_features):
    featureVec = np.zeros(num_features, dtype="float64")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words.split():
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model.wv[word])
    try:
        featureVec = np.divide(featureVec, nwords)
    except RuntimeWarning:
        print(np.divide(featureVec, nwords))
    finally:
        return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float64")
    for review in reviews:
        reviewFeatureVecs[counter] = featureVecMethod(
            review, model, num_features)
        counter = counter+1
    return reviewFeatureVecs


def tf_idf(review):
    with open('tf_idf.pkl', 'rb') as f:
        b_model = pickle.load(f)
    with open('tf_idf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    expected_res = model.predict(b_model.transform([review]).toarray())
    return expected_res[0]


def word_to_vec(review):
    with open('word_to_vec_model.pkl', 'rb') as f:
        b_model = pickle.load(f)
    model = Word2Vec.load('word_to_vec.model')
    expected_res = b_model.predict([featureVecMethod(review, model, 300)])
    return expected_res[0]


def fast_text(review):
    with open('fast_text_model.pkl', 'rb') as f:
        b_model = pickle.load(f)
    model = FastText.load('fast_text/fast_text.model')
    return b_model.predict([featureVecMethod(review, model, 300)])


@app.route('/predictReview')
def predict_review():

    req = request.json
    review_text = req["review"]
    clean_text = text_preprocessing(review_text)
    try:
        model_name = req["model"]
        review = -1
        if 'word_to_vec' in model_name:
            review = word_to_vec(clean_text)
        elif 'tf_idf' in model_name:
            review = tf_idf(clean_text)
        else:
            review = fast_text(clean_text)
    except Exception:
        review = fast_text(clean_text)
    txt_review = 'positive' if review is 1 else 'negative'
    return {'review': txt_review}


if __name__ == '__main__':
    app.run(port=8000)
