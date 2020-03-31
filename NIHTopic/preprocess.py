import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from os import path

import os
import pandas as pd

from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords as sw
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
import pickle
from NIHTopic.utility import mkdir_p
import re


def extract_documents(file_path, file_names):
    files = [each for each in os.listdir(file_path) if
             each.endswith('.txt') and each[:each.index(".txt")] in file_names]
    docs = []
    docs_ID = []
    for file_name in files:
        with open(path.join(file_path, file_name)) as f:
            docID = file_name[:file_name.index('.txt')]
            read_data = f.read()
            first_pos = read_data.find(docID)
            if first_pos >= 0:
                if read_data.find(docID, first_pos + 1) >= 0:
                    read_data = read_data[first_pos + len(docID): read_data.find(docID, first_pos + 1)]
                else:
                    read_data = read_data[first_pos + len(docID):]

            docs.append(read_data)
            docs_ID.append(docID)
            print("preprocess ", docID)
            print("read input data data ", len(read_data))

    return docs, docs_ID

def remove_ordering_number(doc):
    result = []
    for token in doc:
        m = re.search('\d[a-zA-Z]', token)
        if not m:
            result.append(token)

    return result


def remove_numeric(doc):
    result = []
    for token in doc:
        if (len(token) > 1) & (not token.isnumeric()):
            result.append(token)

    return result


def remove_stop_words(doc):
    result = []
    for token in doc:
        if not token in my_stopwords:
            result.append(token)

    return result


def tokenize(docs, tokenizer_engine='nltk'):
    """ Split the documents into tokens.
    :param docs: list of unicode string
    :return:
    """
    tokenized_docs = []
    # init
    if tokenizer_engine == 'spacy':
        print("use spacy")
        import spacy
        spacy.prefer_gpu()  # run on lego
        nlp = spacy.load("en_core_web_sm")
    else:
        tokenizer = RegexpTokenizer(r'\w+')

    for text_input in docs:
        text_input = text_input.lower()  # Convert to lowercase.
        if tokenizer_engine == 'spacy':
            doc = nlp(text_input)
            # tokens = [token.lemma_.strip() for token in doc if (token.lemma_ != '-PRON-') & (token.pos_ != 'PROPN')
            #           & (token.pos_ != 'PUNCT') & (token.pos_ != 'NUM') & (token.pos_ != 'SPACE') & (token.pos_ != 'SYM') & (not token.is_stop)]
            tokens = [token for token in doc if (token.lemma_ != '-PRON-') & (not token.is_stop)]
            tokens = [token.lemma_.strip()for token in tokens if (token.pos_ == 'NOUN') |
                      (token.pos_ == 'VERB')
                      | (token.pos_ == 'ADJ')] # here use lemma_ to avoid past tense of verb
        else:
            tokens = tokenizer.tokenize(text_input)  # Split into words.

        tokens = remove_ordering_number(tokens)
        tokens = remove_numeric(tokens)
        tokens = remove_stop_words(tokens)

        tokenized_docs.append(tokens)

    return tokenized_docs


def run_pipepline(input_docs, tokenizer_engine='nltk'):
    # tokenize docs
    docs = tokenize(input_docs, tokenizer_engine)
    # lemmatize
    # if tokenizer_engine=='nltk':
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
    bigram = Phrases(docs, min_count=20)
    trigram = Phrases(bigram[docs], threshold=60)

    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                # print(token)
                docs[idx].append(token)
                if token =='big_datum':
                    print("stop")

        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                # print(token)
                docs[idx].append(token)


    dictionary = Dictionary(docs)

    # Filter out words that occur less than 10 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=10, no_above=0.6)
    ###############################################################################
    # Finally, we transform the documents to a vectorized form. We simply compute
    # the frequency of each word, including the bigrams.
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return docs, dictionary, corpus


def run_test_pipeline():
    print("rocks :", WordNetLemmatizer().lemmatize("rocks"))
    print("introduced :", WordNetLemmatizer().lemmatize("introduced"))
    text = [
        'Summary \n Aim ~10 1a) We aimed to build a very much 2-layers real-time motinoring system that contain 3.2 32,000 deep learning models dr; A1) We introduced']
    text.extend(['Aim 1: Develop and test statistical learning tool for real-time risk prediction of survival, longitudinal, and multivariate (SLAM) outcome data.'])
    # docs, dictionary, corpus = run_pipepline(text)
    # for doc in docs:
    #     print(doc)
    # print("rocks :", WordNetLemmatizer().lemmatize("rocks"))
    # print("introduced :", WordNetLemmatizer().lemmatize("introduced"))

    docs, dictionary, corpus = run_pipepline(text, tokenizer_engine='nltk')
    for doc in docs:
        print(doc)


import sys, getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"ht:",["tokenizer="])
    except getopt.GetoptError:
        print('preprocess.py -t <tokenizer> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-to':
            print('preprocess.py -t <tokenizer>')
            sys.exit()
        elif opt in ("-t", "--tokenizer"):
            tokenizer = arg

    return tokenizer

if __name__ == '__main__':

    tokenizer = main(sys.argv[1:]).strip()
    print("using:", tokenizer)
    my_stopwords = sw.words('english')
    print(len(my_stopwords))
    # my_stop_words_list = ['abstract', 'abstracting', 'project', 'summary', 'used', 'human', 'approach', 'approaches', 'technique',
    #                       'technology', 'method', 'methods', 'tool', 'algorithm', 'propose', 'proposed', 'using', 'use',
    #                       'study', 'research', 'could', 'would', 'aim', 'want', 'much', 'aa', 'ab', 'dr', 'ac', 'united', 'states',
    #                       'teachers', 'address', 'student', 'tool', 'analysis', 'software', 'new', 'novel', 'design', 'model',
    #                       'patient', 'computer', 'researcher', 'personnel', 'information', 'grant', 'improve','result', 'able', 'allow',
    #                       'health','care', 'include', 'description','specific','identify','applicant','investigator']

    my_stop_words_list = []
    with open('./my_stop_words.txt', 'r') as f:
        for line in f:
            # remove linebreak which is the last character of the string
            my_stop_words_list.append(line[:-1])

    my_stopwords.extend(my_stop_words_list)
    print(len(my_stopwords))

    run_test_pipeline()
    # # get all NIH funded projects
    # df = pd.read_csv("../data/processed_NIH.csv")
    df = pd.read_csv("../data/processed_NIH_ML_DL.csv")
    print(df.head())
    target_files = list(df['Project Number'].unique())
    print(len(target_files))

    # fetch the texts from ML
    docs, docs_IDs = list(extract_documents('../data/federalreporter', target_files))

    # remove filenames that were already fetched
    new_target_files = [each for each in target_files if not each in docs_IDs]
    print("need to fetch {} files from DL folder".format(len(new_target_files)))
    docs_append, docs_append_IDs = list(extract_documents('../data/federalreporter_deep_learning', new_target_files))

    # just for tests
    # docs = docs[:50]
    ###############################################################################
    # So we have a list of 4774 documents, where each document is a Unicode string.
    print(len(docs))
    print(len(docs_append))
    docs.extend(docs_append)
    docs_IDs.extend(docs_append_IDs)
    print('final fetched projects', len(docs))
    print('final fetched projects ID', len(docs_IDs))
    print(docs_IDs[:10])

    docs, dictionary, corpus = run_pipepline(docs, tokenizer_engine=tokenizer)
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    output_path = '../models/corpus_ML_DL_{}'.format(tokenizer)
    mkdir_p(output_path)
    pickle.dump(docs_IDs, open(path.join(output_path, 'corpus_pID.pkl'), 'wb'))
    pickle.dump(corpus, open(path.join(output_path, 'corpus.pkl'), 'wb'))
    dictionary.save(path.join(output_path, 'dictionary.gensim'))
    dictionary.save_as_text(path.join(output_path, 'dictionary.out'))

    # save corpus_ML_DL_nltk_backup text
    output_text_path = '../data/federalreporter_NIH_processed'
    mkdir_p(output_text_path)
    for idx in range(len(docs)):
        text = " ".join(docs[idx])
        with open(path.join(output_text_path, "{}.txt".format(docs_IDs[idx])), 'w+') as f:
            f.write(text)
