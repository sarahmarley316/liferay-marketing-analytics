## Used to topic model free text from Contact Us Form Submissions. This analysis was done in Google Colab.

### Import requirements

#### Import libraries
"""

# Pandas for structuring
import pandas as pd
from pandas import DataFrame

# Numpy for numbers manipulation
import numpy as np

# Statistics for stats 
import statistics as st

# Gensim for LDA models
import gensim
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora, models

# NLTK for word manipulation
import nltk, re, string, collections
from nltk.stem.porter import *
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams # was used to make n-grams in separate code, but not applying heres. n-gram model proved worse.
from collections import Counter #used to obtain frequency for word tokens
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Matplotlib for visiualizations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

"""#### Import dataset"""

# Read CSV in
contactuscomments = pd.read_csv('ContactUsComments10-02-2019.csv')
contactuscomments_original = contactuscomments #creating this in case we need to refer back to the original text document.

# Set index for each document
contactuscomments['index'] = contactuscomments.index
contactuscomments_original['index'] = contactuscomments_original.index

#print to make sure the document came out alright
print(type(contactuscomments))
contactuscomments.sample(3)

"""### Pre-cleaning steps

#### Create custom stopwords list
"""

# Create a custom stopwords list based on domain knowledge and previous exports
nltk_stopwords = stopwords.words('english')
custom_stopwords = frozenset(nltk_stopwords + ['liferay', 'Submitted by Krista Curtis via live chat on LRDC.',
                                               'Submitted', 'LRDC', 'hello','regard', 'like', 'thank','thanks',
                                               'krista','curt','best', 'regard','live','thank','you','curtis',
                                               'grace','cantino','life','ray'])

"""#### Find words with 3 characters or less"""

# Create function to pull words with less than three characters. This returns all words per row with len < 3 words.
def findabbreviations(text):
    len3 = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) <= 3 and token not in gensim.parsing.preprocessing.STOPWORDS and token not in custom_stopwords:
          #print(token)
          len3.append(token)
    return len3

# Create an "abbreviations" variable
abbrevs = contactuscomments['What would you like to talk about?'].map(findabbreviations)

# Finds unique words per row with len <= 3 and concats with a comma. list(set()) returns unique rows.
res = [', '.join(i) for i in abbrevs] 
res = list(set(res)) 

# Converts list to dataframe and labels column name.
dfres = pd.DataFrame(data = res)
dfres.columns = ['A']

# edfres = 'exploded dataframe res'. Using an exploded data frame to fit format requirementws
edfres = pd.DataFrame(dfres['A'].str.split(',').tolist(), index = None).stack()
edfres

"""###### Grab the most common abbreviations or words with len less than or equal to 3"""

## Helpful Hints: the word is called a 'key', and the freq is called the 'values' here. Counter acts as a Dictionary
counter = Counter(edfres)
counter 

counter.most_common(30)

"""#### Abbreviation Cleaning

Create a abbreviation matching list
"""

# Define abbreviation dictionary based on domain knowledge and popular words.
cleanedabbv_dict = {'digital experience platform':'dxp',
                              'content management system': 'cms',
                              'community edition': 'ce',
                              'lr': 'liferay',
                              'request for proposal': 'rfp',
                              'request for information': 'rfi'
                              }

# Find rows that contain the word you're looking for so we can check on it later.
contactuscomments[contactuscomments['What would you like to talk about?'].str.contains('request for proposal')]
print(contactuscomments[1258:1259])

"""Loop through the text to replace words with their counterpart  from list"""

##  "Regex = True" helps us do a partial search. Without it, the code will search the entire value for the exact word from cleanedabbv_dict
for word, initial in cleanedabbv_dict.items():
  contactuscomments['What would you like to talk about?'] = contactuscomments['What would you like to talk about?'].replace(word.lower(),initial,regex=True)

# Test data on row from above to see if replace function worked properly.
print(contactuscomments[1258:1259])

"""#### Let's describe some basic stats about the abbreviations"""

# Grab sum, count, mean, standard deviation, and one standard deviation above the mean.
sumC = sum(counter.values())
countC = len(counter.keys())
mean = sumC/countC
std = st.stdev(counter.values())
std_mean = std + mean
std_mean = round(std_mean,1)

print("The mean frequency of abbreviations or words with a len <= 3 is {}. \nSum = {}, Count = {}. \nThe standard deviation is {}. \nAnything above {} is one std > above the mean.\n".format(round(mean,1),sumC,countC,round(std,1), std_mean))

print(edfres.describe(include = all))

for i in counter.keys():
  if counter[i] > std_mean:
    print(i, counter[i])

"""### Coreference Resolution

#### Install required programs
"""

# Make sure to install these first. Neuralcoref only works with very specific versions. It's picky.
!pip install neuralcoref
!pip install spacy==2.1.0
!python3 -m spacy download en

"""#### Import coref specific libraries"""

# Import spaCy
import spacy
nlp = spacy.load('en')

# Import re and nltk libraries
import re
from nltk.tokenize import sent_tokenize

# Add neuralcoref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp,greedyness=0.5,max_dist=50,blacklist=False)

"""Find example value of coreference"""

pd.set_option('display.max_colwidth', 200)
contactuscomments[contactuscomments['What would you like to talk about?'].str.contains('it')]
contactuscomments[1621:1622]

"""Does it work on some example text that needs coreference resolution?"""

# Print example text and it's type
text = 'I suggest you test forms also with Safari on Mac, I hate Chrome and only use it to test if the form works while using Chrome'
print("Example text:\n",text)
print('\n',"The example text variable is a type:",type(text), '\n')

# Change the type to string so we can run it through the coreference (nlp) function
text = str(text)
doc = nlp(text)
clusters = doc._.coref_clusters
print("The clusters involved in the text variable include ",clusters, '\n')

# Save and print the variable results
resolved_coref = doc._.coref_resolved
print ("New example text after coreference resolution:" )
print(resolved_coref)

"""It works!

#### Coreference for the dataframe time
"""

print(contactuscomments[1621:1622])
a = []
for i in contactuscomments['What would you like to talk about?']:
  i = str(i)
  i = nlp(i)
  resolvedcorefi = i._.coref_resolved
  a.append(resolvedcorefi)

df = pd.DataFrame(data = a, columns = ['What would you like to talk about?'])
print(df[1621:1622])

"""### TextBlob sentiment"""

# Import library
from textblob import TextBlob

textblob_df = TextBlob(str(df['What would you like to talk about?']))
if textblob_df.sentiment.polarity > 0:
  print('The general sentiment of the dataset is positive at a polarity of ',round(textblob_df.sentiment.polarity,2))
  print('The closer the sentiment is to 1, the more positive it is.')
else:
  print('The general sentiment of the dataset is negative at a polarity of ', round(textblob_df.sentiment.polarity,2))
  print('The closer the sentiment is to -1, the more negative it is.')

print('\n')

if textblob_df.sentiment.subjectivity > 0:
  print('The dataset contains feelings of subjectivity at a subjectivity score of ',round(textblob_df.sentiment.subjectivity,2))
  print('The closer the subjectivity score is to 1, the more subjective it is.')
else:
  print('The dataset does not contain feelings of subjectivity at a subjectivity score of ', round(textblob_df.sentiment.subjectivity,2))
  print('The closer the subjectivity score is to -1, the less subjective it is.')

"""### Create lemmatization function"""

# Lemmatize and tokenize the data
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos = 'v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in custom_stopwords and len(token) >= 3:
            result.append(lemmatize_stemming(token))
    return result
  
print(type(df))
df.sample(3)

"""#### Format lemmatized values"""

# Preprocesses data, then saves results as 'processed_docs'
processed_docs = df['What would you like to talk about?'].map(preprocess)
processed_docs.sample(5)
 
df_lemma = processed_docs
df_lemma.sample(3)

"""### Transform dataset and tokenize words"""

# Convert values to list format
df_list = df_lemma.values.tolist()

#The list has a list of lines, and you can't split a "list" of strings.
#You can only split one string at a time. We need to split each line, not the whole list.

# Create word tokens
word_tokens = word_tokenize(str(df_list).strip('[]'))

# The str(contactuscomments).strip('[]') converts the list to a string & removes '[]' around each comment.

"""### Bag of Words

#### Create Dictionary and Corpus - bow
"""

# Create Dictionary
dictionary = Dictionary(df_list)

# Create Corpus
corpus = [dictionary.doc2bow(text) for text in df_list]

# A dictionary contains the unique word tokens in our dataset.
# A corpus is a mapping of [word_id, word_frequency].
  # ie, word_id # 0 takes place 1 time.
  # you can see what word a given word_id corresponds to using: dictionary[0]

# Human readable version of corpus (term-frequency)
[[(dictionary[id],freq) for id, freq in cp] for cp in corpus[:1]]

"""#### Build the LDA topic model variable - bow"""

topic_count = 5

lda_model = gensim.models.LdaMulticore(corpus = corpus,
                                       num_topics = topic_count,
                                       id2word = dictionary,
                                       passes = 10,
                                       iterations = 300,
                                       random_state = 1)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWord: {}'.format(idx,topic))

"""#### Compute model perplexity and coherence score

Helps us measure how good the topic model is.
"""

# Compute Perplexity
print('\nPerplexity: ',lda_model.log_perplexity(corpus)) #lower the better. how good is the model?

# Compute Coherence Score
# Topic Coherence finds the optimal number of topics!!
# We're going to create a series of LDA models with different value k topics and pick the one with the highest coherence score.

coherence_model_lda = CoherenceModel(model = lda_model,
                                     texts = df_list,
                                     dictionary = dictionary,
                                     coherence = 'c_v',
                                    )
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

"""Create the function to compute coherence values"""

def compute_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):   
        model = gensim.models.LdaMulticore(corpus = corpus,
                                               id2word = dictionary,
                                               num_topics = num_topics,
                                               passes = 10,
                                               iterations = 300,
                                               per_word_topics = True,
                                               random_state = 8) 
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model,
                                        texts = df_list,
                                        dictionary = dictionary,
                                        coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

"""Let's plot the coherence scores per number of topics"""

model_list, coherence_values = compute_coherence_values(dictionary = dictionary,
                                                        corpus = corpus,
                                                        texts = df_list,
                                                        start = 2,
                                                        limit = 40,
                                                        step = 6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print(m,"Topics", "has Coherence Value of", round(cv, 4))

"""While these scores are helpful, after considering business needs, we're still only going to take the top 5 results so that we don't have an overbearing amount of topics to go through. This was helpful to know though!

#### Evaluate model on sampled document - bow
"""

# Evaluate performance by classifying sample document using LDA bag of words
# Document in bow_corpus[x] is tested to see which topic it'd fit into best.

pd.set_option('max_colwidth',200)
print(contactuscomments_original['What would you like to talk about?'][10])

for index, score in sorted(lda_model[corpus[10]], key = lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

pd.set_option('max_colwidth',200)
print(contactuscomments_original['What would you like to talk about?'][10])

for idx, topic in lda_model.print_topics(-1):
    print('\nTopic: {}\t \nWord Breakdown: {}'.format(idx, topic))

"""#### Which topic does the sentence belong to? - bow"""

# test model on new document
unseen_document = 'I am interested in purchasing liferay dxp for my company'

bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index,score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Word Weights: {}".format(score, lda_model.print_topic(index, 5)))
    #...(index,5) indicates how many returned word tokens you want to see and their weights

"""#### Display dominant topic of each document - bow"""

def format_topics_sentences(ldamodel = lda_model,
                            corpus = corpus,
                            texts = df_lemma
                           ):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel = lda_model,
                                                  corpus = corpus,
                                                  texts = df_lemma)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(3)

"""#### Display word counts - bow"""

topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in df_lemma for w in w_list] 
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(9,5), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="counter", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    #ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('BoW - Word Count and Importance of Topic Keywords', fontsize=16, y=1.05, color = 'white')    
plt.show()

"""#### PyLDAvis visualization - bow"""

# Import the library and create the pyLDAvis visualization
!pip install pyLDAvis # May need to import this
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, R = 15)
vis

# Saves visualization to Jupyter notebook as an html file
#p = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, R = 15)
#pyLDAvis.save_html(p, 'lda_bow.html')

"""### Term Frequency - Inverse Document Frequency (TF-IDF)

#### Build the TF-IDF topic model variable - tfidf
"""

# Create TF-IDF model and apply transformation
from gensim import corpora, models

tfidf = models.TfidfModel(corpus)
t_corpus = tfidf[corpus]

from pprint import pprint

for doc in t_corpus:
  pprint(doc)
  break

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
topic_count = 5

# Run lda using tf-idf
lda_model_tfidf = gensim.models.LdaMulticore(t_corpus,
                                             num_topics = topic_count,
                                             id2word = dictionary,
                                             passes = 10,
                                             iterations = 300,
                                             random_state = 1)

"""#### Evaluate model on sampled document - tfidf"""

# evaluate performance by classifying sample document using LDA tf-idf.
# document in bow_corpus[x] is tested to see which topic it'd fit into best.
# i.e., for [300], fits the first topic 75% best

pd.set_option('max_colwidth',200)
print(contactuscomments_original['What would you like to talk about?'][10])

for index, score in sorted(lda_model_tfidf[t_corpus[10]], key = lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nWord Breakdown: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWord Breakdown: {}'.format(idx,topic))

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('\nTopic: {} \nWord: {}'.format(idx,topic))

"""#### Which topic does the sentence belong to? - tfidf"""

pd.set_option('max_colwidth',200)
print(contactuscomments_original['What would you like to talk about?'][10])

for idx, score in sorted(lda_model_tfidf[t_corpus[10]], key = lambda tup: -1*tup[1]):
    print("\nTopic: {}  \nScore: {}".format(idx, score, lda_model_tfidf.print_topic(index, 10)))

"""#### Test model on a new document - TF-IDF"""

# Test model on new document
unseen_document = 'I am interested in purchasing liferay dxp for my company'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Word Weights: {}".format(score, lda_model_tfidf.print_topic(index, 4)))
    #...(index,5) indicates how many returned word tokens you want to see and their weights

#test model on new document
unseen_document = 'I am interested in purchasing liferay dxp for my company'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for idx, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Topic: {}\t Score: {}".format(idx, score, lda_model_tfidf.print_topic(index, 4)))
    #...(index,5) indicates how many returned word tokens you want to see and their weights

"""#### Dominant Topic of each document - TF-IDF"""

def format_topics_sentences(ldamodel_tfidf = lda_model_tfidf,
                            corpus = t_corpus,
                            texts = df_lemma
                            ):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel_tfidf[t_corpus]):
        row = row_list[0] if ldamodel_tfidf.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel_tfidf.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel_tfidf = lda_model_tfidf,
                                                  corpus = t_corpus,
                                                  texts=df_lemma
                                                  )

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(100)

"""#### Spit out word counts - TF-IDF"""

# Word Counts of Topic Keywords
from collections import Counter
import matplotlib.colors as mcolors

topics = lda_model_tfidf.show_topics(formatted=False)
data_flat = [w for w_list in df_lemma for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(9,5), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="counter", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    #ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('TF-IDF Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()

"""#### Create pyLDAvis visualization and save to html file - TF-IDF"""

!pip install pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_tfidf, t_corpus, dictionary = dictionary, R = 15)
vis

# saves visualization to Jupyter notebook as an html file
#p = pyLDAvis.gensim.prepare(lda_model_tfidf, t_corpus, dictionary, R = 20)
#pyLDAvis.save_html(p, 'lda_tfidf.html')
