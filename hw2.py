import pickle
from flask import Flask
from flask import url_for, render_template, request, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import nltk
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
morph = MorphAnalyzer()
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from elmo_helpers import tokenize, get_elmo_vectors, load_elmo_embeddings

import logging
logging.basicConfig(filename='application.log', 
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.INFO)

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

app = Flask(__name__)


@app.route('/') #главная
def index():
    return render_template('index.html')

@app.route('/results')
def results():

    with open('corpus_tfidf.pickle', 'rb') as f:
        corpus_tfidf = pickle.load(f)
        
    with open('docs_id.pickle', 'rb') as f:
        docs_id = pickle.load(f)
        print(len(docs_id))
        
    with open('bm25_matrix.pickle', 'rb') as f:
        bm25_matrix = pickle.load(f)
        
    with open('position_of_word.pickle', 'rb') as f:
        position_of_word = pickle.load(f)
    
    with open('lemmas_vocab.pickle', 'rb') as f:
        lemmas_vocab = pickle.load(f)
    
    with open('docs_vectors.pickle', 'rb') as f:
        docs_vectors = pickle.load(f)
    
    with open('Indexed_ELMO.pickle', 'rb') as f:
        indexed, id_to_text, query_to_dupl_id = pickle.load(f)
        #print('ok')
    
    if request.args:
        try:
            query = request.args['query']
            model_name = request.args['model']
            logging.info(f'Query: {query}, Model: {model_name}')
            
            if model_name == 'TF-IDF':
                vec = TfidfVectorizer()
                res_for_print = tf_idf(query, vec, corpus_tfidf, docs_id)
            
            if model_name == 'BM25':
                res_for_print = bm25(query, bm25_matrix, position_of_word, docs_id)
            
            if model_name == 'FastText':
                fast_model = 'fasttext/model.model'
                fasttext_model = KeyedVectors.load(fast_model)
                res_for_print = fasttext(query, fasttext_model, lemmas_vocab, docs_vectors, docs_id)
                
            if model_name == 'ELMO':
                #print('ok!')
                tf.reset_default_graph()
                #print('ok!')
                elmo_path = 'ELMO'
                print(elmo_path)
                batcher, sentence_character_ids, elmo_sentence_input = load_elmo_embeddings(elmo_path)
                print('ok')
                res_for_print = emlo(query, batcher, sentence_character_ids, elmo_sentence_input, indexed, docs_id)
            
            logging.info(f'Result: {res_for_print}')
            
            return render_template('results.html', query = query, result = res_for_print)
        
        except Exception as e:
            logging.exception(f'Error with query {query} and model {model_name}:\n\n repr(e)')
    
    
def preprocessing(text):
    normal_forms = []
    tokenizer = RegexpTokenizer(r'\w+')
    
    tokens = [morph.parse(token)[0] for token in tokenizer.tokenize(text)]
    normal_forms = [token.normal_form for token in tokens if token not in russian_stopwords and token.normal_form != 'это']
    preproc_text = ' '.join(normal_forms)
       
    return preproc_text

def preprocessing_FastText(text):
    
    text = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"\ufeff]', r'', str(text))
    text = WordPunctTokenizer().tokenize(text)
    preproc_text = ''
    tokens = [morph.parse(token)[0] for token in text]
    normal_forms = [token.normal_form for token in tokens if token not in russian_stopwords and token.normal_form != 'это']
    preproc_text = ' '.join(normal_forms)
       
    return preproc_text


def tf_idf(query, vec, corpus, docs_id):
    
    res = {}
    query_preproc = preprocessing(query)
    
    matrix = vec.fit_transform(corpus)
    
    vec_query = vec.transform([query_preproc])

    res_vec = matrix.dot(vec_query.transpose())
    results_df = pd.DataFrame(res_vec.toarray())
    
    for doc_id, tf_idf in enumerate(results_df[0]):
        
        if doc_id == len(results_df[0]) - 1:
            break
        
        res[docs_id[str(doc_id)]] = round(tf_idf, 2)
          
    result = sorted(res.items(), key=lambda x: (x[1],x[0]), reverse=True)

    return result[:10]

def bm25(query, matrix, position_of_word, docs_id):
    
    res = {}
    query_preproc = preprocessing(query)
    vec_query = np.zeros((matrix.shape[1], 1))
    
    for q in query_preproc.split():
        
        try:
            vec_query[position_of_word[q]] = 1
            
        except KeyError:
            pass
        
    res_vec = matrix.dot(vec_query)
    
    for doc_id, bm_25 in enumerate(res_vec):
        
        try:
            res[docs_id[str(doc_id)]] = round(bm_25[0], 2)
        
        except KeyError:
            pass
    
    result = sorted(res.items(), key=lambda x: (x[1],x[0]), reverse=True)  
    
    return result[:10]


def sent_vectorizer(sent, model, lemmas_vectors): # превращает каждое слово в вектор
   
    if type(sent) != str:
        
        sent_vector = np.zeros((model.vector_size,))
        return sent_vector
    
    sent = sent.split()
    lemmas_vectors = np.zeros((len(sent), model.vector_size))
    
    for idx, lemma in enumerate(sent):
        if lemma in model.vocab:
            lemmas_vectors[idx] = model[lemma]
    
    sent_vector = lemmas_vectors.sum(axis=0) # суммируем вектор для каждого документа
    return sent_vector
 

def cos_similarity(docs_vectors, query_vec):
  cos_sims = {} # словарь косинусных близостей запроса к каждому доку
  
  for i, vector in enumerate(docs_vectors['q1_vector']):
    cos_sim = cosine_similarity(vector.reshape(1, -1), query_vec.reshape(1, -1))
    cos_sims[i] = cos_sim[0][0]
  
  return cos_sims

def fasttext(query, fasttext_model, lemmas, docs_vectors, docs_id):
    
    result = {}
    
    query_lemm = preprocessing_FastText(query)
    
    query_vec = sent_vectorizer(query_lemm, fasttext_model, lemmas)
    
    cos_sims = cos_similarity(docs_vectors, query_vec)
    
    for c in cos_sims.keys():
        try:
            result[docs_id[str(c)]] = round(cos_sims[c], 2)
        except KeyError:
            pass
        
    res = sorted(result.items(), key=lambda x: (x[1],x[0]), reverse=True)
    
    return res[:10]

def cos_sim(v1, v2):
    """Counts cosine similarity between two vectors"""
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def crop_vec(vect, sent):
    """
    Crops dummy values

    :param vect: np.array, vector from ELMo
    :param sent: list of str, tokenized sentence
    :return: np.array

    """
    cropped_vector = vect[:len(sent), :]
    cropped_vector = np.mean(cropped_vector, axis=0)
    return cropped_vector

def prepare_query(query, batcher, sentence_character_ids, elmo_sentence_input):
    """ 
    Gets vector of query

    :param query: str
    :param batcher, sentence_character_ids, elmo_sentence_input: ELMo model
    
    :return: vector of query
    """
    q = [tokenize(query)]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vector = crop_vec(get_elmo_vectors(sess, q, batcher,
                                           sentence_character_ids,
                                           elmo_sentence_input)[0], q[0])
    return vector

def emlo(query, batcher, sentence_character_ids,
                     elmo_sentence_input, indexed, docs_id):
    """
    Search query in corpus

    :param: query: str
    :param batcher, sentence_character_ids, elmo_sentence_input: ELMo model
    :param indexed: np.array, matrix of indexed corpus

    :return: list, sorted results
    """
    q = prepare_query(query, batcher, sentence_character_ids, 
                      elmo_sentence_input)

    result = {}
    for i, doc_vector in enumerate(indexed):
        score =  cos_sim(q, doc_vector)
        if type(score) is np.float32:
            try:
                result[docs_id[str(i)]] = score
            except KeyError:
                pass
    
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]

if __name__ == '__main__':
    app.run(debug=False)

