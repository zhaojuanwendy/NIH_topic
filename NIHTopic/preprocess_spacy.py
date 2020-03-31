import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import spacy


# def lemmatizer(doc):
#     # This takes in a doc of tokens from the NER and lemmatizes them.
#     # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
#     doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
#     doc = u' '.join(doc)
#     return nlp.make_doc(doc)
#
#
# def remove_stopwords(doc):
#     # This will remove stopwords and punctuation.
#     # Use token.text to return strings, which we'll need for Gensim.
#     doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
#     return doc

def tokenize(texts, max_length, skip=-2, merge=False, **kwargs):
    """ Uses spaCy to quickly tokenize text and return an array
    of indices.
    This method stores a global NLP directory in memory, and takes
    up to a minute to run for the time. Later calls will have the
    tokenizer in memory.
    Parameters
    ----------
    text : list of unicode strings
        These are the input documents. There can be multiple sentences per
        item in the list.
    max_length : int
        This is the maximum number of words per document. If the document is
        shorter then this number it will be padded to this length.
  """
    nlp = spacy.load("en_core_web_sm")
    my_stop_word_list = ['abstract', 'project', 'summary', 'used', 'human', 'approach', 'approaches', 'technique',
                         'technology',
                         'method', 'methods', 'algorithm', 'propose', 'proposed', 'using', 'use', 'study',
                         'could', 'would', 'aim', 'want', 'much', 'aa', 'ab', 'dr', 'ad', 'ac', 'united', 'states',
                         'teachers']
    nlp.Defaults.stop_words.update(my_stop_word_list)
    # The add_pipe function appends our functions to the default pipeline.
    text_strip = text[0].replace("\n", '')
    doc = nlp(text_strip)
    from pprint import pprint
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    pprint(" ".join([token.text for token in doc]))
    tokens = [token.lemma_.strip() for token in doc if (token.lemma_ != '-PRON-')
              & (token.pos_ != 'PROPN') & (token.pos_ != 'PUNCT') & (token.pos_ != 'NUM') & (token.pos_ != 'SPACE') & (token.pos_ != 'SYM') & (not token.is_stop)]
    # sentence = """Following mice attacks, caring farmers were marching to Delhi for better living conditions.
    # Delhi police on Tuesday fired water cannons and teargas shells at protesting farmers as they tried to
    # break barricades with their cars, automobiles and tractors. We introduced a paper"""
    # doc = nlp(sentence)
    # tokens = [token.lemma_ for token in doc]
    pprint(" ".join([token for token in tokens]))
    # doc = nlp(text[0])
    # spacy.displacy.render(doc, style='ent', jupyter=True)



if __name__ == "__main__":

    text = ['Summary \n Aim --1a) We aimed to build a 2-layers real-time machine learning models dr; Gene-Modified; ']
    # text = ['Peptide hormones regulate embryonic development and most physiological processes r']
    tokenize(text, 5)


