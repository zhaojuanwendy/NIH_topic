
from os import path
import pickle
import matplotlib.pyplot as plt

# Train LDA model.
from gensim.models import LdaModel, TfidfModel
from gensim.corpora import Dictionary
from pprint import pprint

from utility import mkdir_p


# def visualize_topic(topic_word_fequency, feature_names, image_path, n_components):
#
#     for topic_index in np.arange(n_components):
#         word_frequency_per_topic = topic_word_fequency[topic_index, :]
#         word_frequency_dic = dict(zip(feature_names, word_frequency_per_topic))
#         word_cloud = WordCloud(width=485, height=280, max_words=60, colormap='plasma',
#                                # ackground_color="rgba(255, 255, 255, 0)", mode="RGBA")
#                                ).generate_from_frequencies(word_frequency_dic)
#         #     word_cloud.recolor(color_func=image_colors)
#         word_cloud.to_file(path.join(image_path, 'Topic_T{}.png'.format(topic_index)))

def lda(corpus, num_topics=10, passes=20, model='count'):
    # Set training parameters.
    chunksize = 2000
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    if model =='tfidf':
        tfidf = TfidfModel(corpus, smartirs='ntc')
        for doc in corpus:
            print([[dictionary[id], freq] for id, freq in doc])
        corpus = tfidf[corpus]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every,
        random_state=2
    )

    ###############################################################################
    # We can compute the topic coherence of each topic. Below we display the
    # average topic coherence and print the topics in order of topic coherence.
    #
    # Note that we use the "Umass" topic coherence measure here (see
    # :py:func:`gensim.models.ldamodel.LdaModel.top_topics`), Gensim has recently
    # obtained an implementation of the "AKSW" topic coherence measure (see
    # accompanying blog post, http://rare-technologies.com/what-is-topic-coherence/).
    #
    # If you are familiar with the subject of the articles in this dataset, you can
    # see that the topics below make a lot of sense. However, they are not without
    # flaws. We can see that there is substantial overlap between some topics,
    # others are hard to interpret, and most of them have at least some terms that
    # seem out of place. If you were able to do better, feel free to share your
    # methods on the blog at http://rare-technologies.com/lda-training-tips/ !
    #

    top_topics = lda_model.top_topics(corpus)  # run lda
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    log_perplexity = lda_model.log_perplexity(corpus)
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)

    return lda_model, avg_topic_coherence, log_perplexity

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

tokenizer = main(sys.argv[1:])
model_name ='count'
print("using:", tokenizer)
# load data
print("loading corpus processed by ", tokenizer)
docs_PID = pickle.load(open(path.join('../models/corpus_ML_DL_{}/corpus_pID.pkl').format(tokenizer), 'rb'))
corpus = pickle.load(open(path.join('../models/corpus_ML_DL_{}/corpus.pkl').format(tokenizer), 'rb'))
dictionary = Dictionary.load('../models/corpus_ML_DL_{}/dictionary.gensim'.format(tokenizer))

# mkdir_p('../models/ML_DL_{}_{}'.format(tokenizer,model_name))
mkdir_p('../models/ML_DL_{}'.format(tokenizer))
avg_topic_coherences = []
log_perplexities = []
num_topics_range = range(5, 60, 5)
for num_topics in num_topics_range:
    # run lda model
    lda_model, avg_topic_coherence, log_perplexity = lda(corpus, num_topics, model=model_name)
    avg_topic_coherences.append(avg_topic_coherence)
    log_perplexities.append(log_perplexity)

    # lda_model.save('../models/ML_DL_{}_{}/lda-{}.model'.format(tokenizer, model_name, num_topics))
    lda_model.save('../models/ML_DL_{}/lda-{}.model'.format(tokenizer, num_topics))

import seaborn as sns
ax = sns.lineplot(x=num_topics_range, y=avg_topic_coherences)
plt.xlabel("topics number")
plt.ylabel("coherence")
plt.xticks(num_topics_range)
plt.savefig('../images/coherence_{}_{}.png'.format(tokenizer, model_name))

plt.figure()
ax = sns.lineplot(x=num_topics_range, y=log_perplexities)
plt.xlabel("topics number")
plt.ylabel("log_perplexity")
plt.xticks(num_topics_range)
plt.savefig('../images/log_perplexity_{}_{}.png'.format(tokenizer, model_name))





# import matplotlib.colors as mcolors
# cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
#
# cloud = WordCloud(background_color='white',
#                   width=2500,
#                   height=1800,
#                   max_words=30,
#                   colormap='tab10',
#                   color_func=lambda *args, **kwargs: cols[i],
#                   prefer_horizontal=1.0)
#
# topics = lda_model.show_topics(formatted=False, num_topics=num_topics)
#
# fig, axes = plt.subplots(round(num_topics/2), 2, figsize=(10, 10), sharex=True, sharey=True)
#
# for i, ax in enumerate(axes.flatten()):
#     print(i)
#     fig.add_subplot(ax)
#     topic_words = dict(topics[i][1])
#     cloud.generate_from_frequencies(topic_words, max_font_size=300)
#     plt.gca().imshow(cloud)
#     plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
#     plt.gca().axis('off')
#
#
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.axis('off')
# plt.margins(x=0, y=0)
# plt.tight_layout()
#
# mkdir_p('../images')
# plt.savefig('../images/word_cloud_topics{}.png'.format(num_topics), dpi=720)
# plt.show()



# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# pyLDAvis.enable_notebook()
