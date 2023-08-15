from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from nltk.stem import PorterStemmer
import nltk
from nltk.stem import 	WordNetLemmatizer

class textProcessing:

  def __init__(self,articles=[]):
    self.listStopWords=stopwords.words('english')
    self.garbage=['This is an open access article, free of all copyright, and may be freely reproduced, distributed, transmitted, modified, built upon, or otherwise used by anyone for any lawful purpose','The work is made available under the Creative Commons CC0 public domain dedication.','Elsevier B.V.','	©','Elsevier LtdA','by the authors. Licensee MDPI, Basel, Switzerland.Abstract: ','Springer Nature Singapore Pte Ltd. ']

    if(len(articles)!=0):
      self.size=len(articles)
      self.listArticles=articles
      self.createAbstractList()
    

  def createAbstractList(self):
    self.listAbstracts=[]
    for article in self.listArticles:
      keywords=""
      for keyword in article.keywords:
        keywords=keywords+" "+keyword
      self.listAbstracts.append(self.cleanAbstract(article.abstract+" "+article.title+" "+keywords))

  def cleanAbstract(self,abstract):
    for item in self.garbage:
      abstract=abstract.replace(item,'')
    abstract=abstract.lower()
    abstract_tokens=word_tokenize(abstract)
    abstract_without_sw = [word for word in abstract_tokens if not word in self.listStopWords]
    abstract_without_sw = [word for word in abstract_without_sw if word.isalpha()]
    ps = PorterStemmer()
    abstract_without_sw = [ps.stem(word) for word in abstract_without_sw]
    # wordnet_lemmatizer = WordNetLemmatizer()
    # abstract_without_sw=[wordnet_lemmatizer.lemmatize(word) for word in abstract_without_sw]
    abstract=(" ").join(abstract_without_sw)
    return abstract
  def preProcesingListData(self, listData):
    newData=[]
    for item in listData:
      newData.append(self.cleanAbstract(item))
    return newData
  def cleanAbstract2(self,abstract):
    for item in self.garbage:
      abstract=abstract.replace(item,'')
    abstract=abstract.lower()
    abstract_tokens=word_tokenize(abstract)
    abstract_without_sw = [word for word in abstract_tokens if not word in self.listStopWords]
    abstract_without_sw = [word for word in abstract_without_sw if word.isalpha()]
    abstract=(" ").join(abstract_without_sw)
    return abstract

  def createTfIDF(self):
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), norm=None)
    vectors = vectorizer.fit_transform(self.listAbstracts)
    self.dense=vectors.todense()
  def createTfIDF(self, data):
    self.vectorizer = TfidfVectorizer(ngram_range=(1, 1),norm=None)
    return self.vectorizer.fit_transform(data)
  
  def tfidf(self,data,ngram_range=(1, 3)):
    min=int(0.01*len(data))
    min = 2 if min<2 else min
    min = 10 if min>10 else min
    max=0.80
    self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english',use_idf=True,norm=None, max_df=max, min_df=min)
    tfidf_matrix = self.vectorizer.fit_transform(data)
    return tfidf_matrix
  def tfidf_for_labels(self,data,ngram_range=(3, 3)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words='english',use_idf=True,norm=None)
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer
    
  def createDistanceOneToMany(self):
    self.distances=np.zeros(self.size,dtype=float)
    for j in range(1,self.size):
      self.distances[j]=cosine_similarity(np.array(self.dense[0]), np.array(self.dense[j]))
    data=[[item] for item in self.distances]
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(data)
    self.distances=np.array([item[0] for item in mm_data])
    
  def createDistances(self):
    self.distances=np.zeros((self.size,self.size),dtype=float)
    for i in range(0,self.size):
      for j in range(i,self.size):
        if(i!=j):
          self.distances[i][j]=cosine_similarity(np.array(self.dense[i]), np.array(self.dense[j]))
          self.distances[j][i]=self.distances[i][j]
      data=[[item] for item in self.distances[i]]
      mm = MinMaxScaler()
      mm_data = mm.fit_transform(data)
      self.distances[i]=np.array([item[0] for item in mm_data])
      self.distances[i][i]=1
      
