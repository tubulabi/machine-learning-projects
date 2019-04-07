#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd 


# In[35]:


top_10_Sender = pd.read_csv('C:\\Users\\ce1059\\Vmware\\enron_cleaned_sent_emails.csv', error_bad_lines=False)


# In[36]:


top_10_Sender.head()


# In[63]:


top_10_Sender['Name'] = top_10_Sender['file'].str.split("/",n=1,expand=True)[0]
top_10_Sender.head()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


sender_count = top_10_Sender['Name'].value_counts()
sender_count = sender_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(sender_count.index , sender_count.values, alpha = 0.8)
plt.title("Top 10 Email Sender")
plt.ylabel('No of Time Each Sender sent mail', fontsize = 12)
plt.xlabel('Sender code', fontsize = 12)
plt.xticks(rotation=70)
plt.show()


# In[6]:


import pandas as pd

email = pd.read_csv('C:\\Users\\ce1059\\Vmware\\enron_cleaned_sent_emails.csv', error_bad_lines=False)
email_data = email[['body']]
email_data ['index'] = email_data.index
documents = email_data 


# In[7]:


print(len(documents))
print(documents[:5])


# In[8]:


print(documents.shape)
documents = documents.dropna()
documents.count()


# In[9]:


#Basic Data Cleaning 
###Convert to Lower Case
documents['body']= documents['body'].apply(lambda body: body.strip().lower())
documents.head()


# In[10]:


#Removing URLs
documents['body'] = documents['body'].str.replace('http\S+|www.\S+', 'URL', case=False)


# In[11]:


#Remove special character, Punctuations and numbers
documents['body'] = documents['body'].str.replace("[^a-zA-Z#]", " ")
documents.head()


# In[12]:


#Loading gensim and nltk libraries
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')


# In[13]:


stemmer = SnowballStemmer('english')


# In[9]:


##Words reduced to the original word
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


#Remove more than two space with one space
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[10]:


doc_sample = documents[documents['index'] == 10000].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[11]:


processed_docs = documents['body'].map(preprocess)


# In[12]:


processed_docs.head(20)


# In[13]:


#Bag of Words (Top 10 words)
top_10_words = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in top_10_words.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[22]:


#keep only the first 100000 most frequent tokens
top_10_words.filter_extremes(no_below=8, no_above=0.5, keep_n=100000)


# In[23]:


bow_corpus = [top_10_words.doc2bow(doc) for doc in processed_docs]
bow_corpus[10000]


# In[24]:


#Analyse the corpus
bow_doc_10000 = bow_corpus[10000]
for i in range(len(bow_doc_10000)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_10000[i][0], 
                                               top_10_words[bow_doc_10000[i][0]], 
bow_doc_10000[i][1]))


# In[21]:


#Creating TF-IDF model
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[25]:


#Running LDA using Bag of Words
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=top_10_words, passes=2, workers=2)


# In[26]:


#For each topic, we will explore the words occuring in that topic and its relative weight
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[27]:


#Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=top_10_words, passes=2, workers=4)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[28]:


processed_docs[10000]


# In[29]:


#Performance evaluation by classifying sample document using LDA Bag of Words model
for index, score in sorted(lda_model[bow_corpus[10000]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# In[30]:


#Performance evaluation by classifying sample document using LDA TF-IDF model
for index, score in sorted(lda_model_tfidf[bow_corpus[10000]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# In[1]:


print("TF-IDF model is giving better accuracy as compared to Bag of word model")
print("In TF-IDF model the score in 1st topic is 45% where as in Bag of word model score in 1st topic is 28% ")


# In[31]:


#Testing model on unseen document
test_document = 'Meeting at 10:30'
bow_vector = top_10_words.doc2bow(preprocess(test_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


# In[36]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


# In[40]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[41]:


cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# In[42]:




