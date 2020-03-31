import pandas as pd

from os import path, listdir
from gensim.corpora import Dictionary
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def extract_documents(file_path, file_names):
    files = [each for each in listdir(file_path) if each.endswith('.txt') and each[:each.index(".txt")] in file_names]
    docs = []
    # docs_fn = []
    for file_name in files:
        with open(path.join(file_path, file_name)) as f:
            read_data = f.read()
            docs.append(read_data)
            # docs_fn.append(file_name)

    return docs



def preprocessDocuments(docs):
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    searched_words = 0
    for idx in range(len(docs)):

        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = docs[idx][docs[idx].index('\n') + 1:]  # skip first line
        if (docs[idx].find('machine learning')):
            searched_words += 1

        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    # Remove stop words
    my_stopwords = sw.words('english')
    my_stopwords.extend(['abstract', 'project', 'summary', 'used', 'human'])
    docs = [[token for token in doc if token not in my_stopwords] for doc in docs]



    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    from gensim.models import Phrases

    # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
    bigram = Phrases(docs, min_count=10)
    trigram = Phrases(bigram[docs], threshold=100)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                # print(token)
                docs[idx].append(token)

        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                # print(token)
                docs[idx].append(token)
    return docs

df = pd.read_csv("../data/processed_NIH.csv")
print(df.head())
df['Fiscal Year'] = df['Fiscal Year'].astype(int)
print(df.dtypes)
print(sorted(list(df['Fiscal Year'].unique())))
data_root_path ='../data/federalreporter_NIH_processed'

new_docs =[]

for agency in ['NIH']:  # sorted(list(df['Agency'].unique())):
    for year in [2018]: #sorted(list(df['Fiscal Year'].unique())):
        project_numbers = list(df[(df['Fiscal Year'] == year) & (df['Agency'] == agency)]['Project Number'].unique())
        print(year, agency, len(project_numbers))
        docs = []
        if len(project_numbers) != 0:
            docs = extract_documents(data_root_path, project_numbers)
            new_docs.append(" ".join(docs))  # merge all docs for an agency within a year to a new doc
        else:
            new_docs.append("")

            # store in a dict with key = (year, agency)
        # print(y, agency)
# create a unit dictionary cross documents
dictionary = Dictionary(new_docs)
corpus = []

for i, agency in enumerate(['NIH']): # sorted(list(df['Agency'].unique())):
    for j, year in enumerate([2018]): #sorted(list(df['Fiscal Year'].unique())):
        doc = new_docs[i*len([2018])+j]
        # dictionary.filter_extremes(no_below=10, no_above=0.8)
        corpus.append(dictionary.doc2bow(doc))
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))
        for doc in corpus:
            print([[dictionary[id], freq] for id, freq in doc])





