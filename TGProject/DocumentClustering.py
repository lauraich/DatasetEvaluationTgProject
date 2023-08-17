# Tratamiento de datos
# ==============================================================================
import json
import math
import random
import numpy as np
import pandas as pd
import numpy.random
import requests

# Gráficos
# ==============================================================================

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from collections import Counter

# Preprocesado y modelado
# ==============================================================================
from sklearn.preprocessing import Normalizer, StandardScaler, scale

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import GaussianMixture

# Configuración warnings
# ==============================================================================
import warnings

from SklearnAlgorithms.KmeansCosine import KMeans
from SklearnAlgorithms.PairwiseDegree import cosine_distances
from SklearnAlgorithms.SilhouetteCosine import davies_bouldin_score, silhouette_score
from SklearnAlgorithms.SpectralClusteringCosine import SpectralClustering, spectral_clustering

warnings.filterwarnings('ignore')
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification

import skfuzzy as fuzz

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

import re

class DocumentClustering:
    def __init__(self,data,maxClusters=15) -> None:
        #La Data es un TFIDF como array
        self.data=data
        self.maxClusters=maxClusters
    def calculateNumClusters(self,method="bic"):
        match method:
            case "bic": 
                return self.BICScore()
            case "silhouette": 
                return self.numClustersXSilhouette()
            case "davies-bouldien":
                return self.daviesBouldin()
    def numClustersXSilhouette(self)->int:
        vectorCoeficientesxClusters=np.zeros(shape=self.maxClusters-1,dtype=float)
        lsa = make_pipeline(TruncatedSVD(n_components=100))
        data_lsa = lsa.fit_transform(self.data.toarray())
        """ pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.data.toarray())
        data_lsa=X_pca """
        numDocumentos=len(data_lsa)
        if(numDocumentos>self.maxClusters):
            numDocumentos=self.maxClusters+1
        for i in range(2,numDocumentos):
            kmeans = KMeans(
                    n_clusters=i,
                    max_iter=100,
                    n_init=5,
                    random_state=i,
                ).fit(data_lsa)
            labels = kmeans.labels_
            sc=silhouette_score(data_lsa, labels, metric="cosine")
            vectorCoeficientesxClusters[i-2]=sc  
        index=np.where(vectorCoeficientesxClusters==vectorCoeficientesxClusters.max())[0][0]
        return index+2
    
    def daviesBouldin(self)->int:
        vectorCoeficientesxClusters=np.zeros(shape=self.maxClusters-1,dtype=float)
        lsa = make_pipeline(TruncatedSVD(n_components=100))
        data_lsa = lsa.fit_transform(self.data.toarray())
        numDocumentos=len(data_lsa)
        if(numDocumentos>self.maxClusters):
            numDocumentos=self.maxClusters+1
        for i in range(2,numDocumentos):
            kmeans = KMeans(
                    n_clusters=i,
                    max_iter=100,
                    n_init=5,
                    random_state=i,
                ).fit(data_lsa)
            labels = kmeans.labels_
            sc=davies_bouldin_score(data_lsa, labels)
            vectorCoeficientesxClusters[i-2]=sc  
        index=np.where(vectorCoeficientesxClusters==vectorCoeficientesxClusters.min())[0][0]
        return index+2
    def BICScore(self)->int:
        lsa = make_pipeline(TruncatedSVD(n_components=100))
        data_lsa = lsa.fit_transform(self.data.toarray())
        numDocumentos=len(data_lsa)
        upprBound=self.maxClusters 
        if(upprBound>numDocumentos):
            upprBound=numDocumentos
        n_components = range(2, upprBound)
        covariance_type = ['spherical', 'tied', 'diag', 'full']
        score=[]
        for cov in covariance_type:
            for n_comp in n_components:
                gmm=GaussianMixture(n_components=n_comp,covariance_type=cov)
                gmm.fit(data_lsa)
                score.append((cov,n_comp,gmm.bic(data_lsa)))
        min=score[0][2]
        numClusters=score[0][1]
        for scor in score:
            if(scor[2]<min):
                min=scor[2]
                numClusters=scor[1]
        return numClusters
    def clusteringKMeans(self, maxIterations=1000, numeroClusters=None):
        silhouetteScore=-100000
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters
        """   pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.data.toarray())
        X_lsa=X_pca
        
        tsne = TSNE(n_components=1)
        X_tsne = tsne.fit_transform(self.data.toarray())
        X_lsa=X_tsne """
        lsa = make_pipeline(TruncatedSVD(n_components=5))
        X_lsa = lsa.fit_transform(self.data.toarray())
        #X_lsa=self.data.toarray()
        sed=np.random.randint(10,self.numClusters+10)
        for seed in range(2,12):
            sed=np.random.randint(10,self.numClusters+10)
            kmeans = KMeans(
                        # tol=0.01,
                        #verbose=50,
                        n_clusters=self.numClusters,
                        max_iter=maxIterations,
                        init='k-means++'
                        #n_init=sed
                        #random_state=sed
                    ).fit(X_lsa)
            labels = kmeans.labels_
            ratio=self.SSE(kmeans.cluster_centers_, X_lsa, labels)
            ss=silhouette_score(X_lsa, labels)
            
            if(ss>silhouetteScore): 
                    silhouetteScore=ss
                    self.kmeansClusterResult=kmeans
        return X_lsa
    def SEE(self,centroides, x, labels):
        see=0
        sumaDistanciasAlCuadrado=0
        for label in np.unique(labels):
            for j in np.where(labels==label):
                j=j[0]
                dist=cosine_distances([centroides[label]], [x[j]])
                sumaDistanciasAlCuadrado+=dist*dist
        centroidedetodoslosecentroides=centroides.mean(axis=0)
        distanciaentrecentroides=0
        for label in np.unique(labels):
            dist=cosine_distances([centroides[label]],[centroidedetodoslosecentroides])
            distanciaentrecentroides+=dist*dist
        ratio=distanciaentrecentroides/(distanciaentrecentroides+sumaDistanciasAlCuadrado)
        return ratio
    def SSE(self,centroids, data, labels):
        sse = 0
        for i in range(data.shape[0]):
            centroid = centroids[labels[i]]
            
            squared_error = np.sum((data[i] - centroid) ** 2)
            sse += squared_error
        return sse
    def clusteringSpectral(self, numeroClusters=None):
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters
        n_components=150 if self.data.shape[1]>=150 else self.data.shape[1]
        lsa = make_pipeline(TruncatedSVD(n_components=n_components))
        X_lsa = lsa.fit_transform(self.data.toarray())
        
        silhouetteScore=-1000
        #for seed in range(0,15):
        r=random.Random()
        sed=r.randrange(start=2, stop=self.numClusters)
        spectralClusterResult = SpectralClustering(
                        n_clusters=self.numClusters,
                        eigen_solver="arpack",
                        affinity="cosine", assign_labels='cluster_qr'
                        ).fit(X_lsa,metric="cosine")
        labels = spectralClusterResult.labels_
            #ss=silhouette_score(X_lsa, labels)
        """if(ss>silhouetteScore):
                    print("semilla: ",sed)
                    silhouetteScore=ss"""
        
        self.spectralClusterResult=spectralClusterResult
        return X_lsa,lsa
                    
    def clusteringFuzzyCMeans(self, numDocs, numeroClusters=None):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(self.data.toarray())
        lsa = TruncatedSVD(2, algorithm = 'arpack')
        dtm_lsa = lsa.fit_transform(X_std)
        #dtm_lsa=self.data
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters
        a= pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
        alldata = np.vstack((a['component_1'], a['component_2']))
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, self.numClusters, m=2, error=0.005, maxiter=1000, init=None, metric="cosine")
        matrixmembresia=pd.DataFrame(u)
        matrixmembresia.to_csv("mmembresia.csv")
        n=numDocs
        c=self.numClusters
        
        tope=self.overlappingFuzzyCmeans(u)
        newlabels, co=self.countOverlappingsxGroup(u,tope)  
        
        #Groups es una matriz de numDocumentos*numClusters donde cada fila es un documento y cada columna dentro 
        #de la fila es el grado de pertenencia del documento a un grupo
        return newlabels,
    
    def clusteringFuzzyCMeansV2(self, numDocs, numeroClusters=None):
        """
        Ésta version usa la matriz de pertenencia para hacer un cubrimiento en su vertical del 95% de pertenencia

        Args:
            numDocs (_type_): Número de documentos

        Returns:
            _type_: _description_
        """
        scaler = StandardScaler()
        X_std = scaler.fit_transform(self.data.toarray())
        #X_std=self.data.toarray()
        lsa = TruncatedSVD(2, algorithm = 'arpack')
        #lsa = make_pipeline(TruncatedSVD(2, algorithm = 'arpack'))
        dtm_lsa = lsa.fit_transform(X_std)
        
        #dtm_lsa=self.data
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters
        a= pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
        alldata = np.vstack((a['component_1'], a['component_2']))
        
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, self.numClusters, m=2, error=0.005, maxiter=1000, init=None, metric=cosine_distances)
        n=numDocs
        c=self.numClusters
        newlabels=[]
        co=0
        for doc in range(numDocs):
            listaPertenencia=[(cluster,u[cluster][doc]) for cluster in range(self.numClusters)]
            listaPertenencia=sorted(listaPertenencia, key=lambda x: x[1], reverse=True)
            pertenenciaAcumulada=0
            labels=[]
            for i in listaPertenencia:
                if(pertenenciaAcumulada<0.95):
                    labels.append(i[0])
                    pertenenciaAcumulada+=i[1]
                else:
                    break
            newlabels.append(labels)
            label_pred=[]
            for lbl in newlabels:
                label_pred.append(lbl[0])
            if(len(labels)>1):
                co+=1  
        return newlabels,co,u
    def getLabelsClusters(self,lsa, centroides, vectorizer):
        original_space_centroids = lsa[0].inverse_transform(centroides)
        #centroides=np.array(centroides)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        labels=[]
        for i in range(self.numClusters):
            wordList=[]
            for ind in order_centroids[i, :10]:
                wordList.append(terms[ind])
            labels.append(wordList)
        return labels
    


    def get_common_phrases(self,texts, maximum_length=3, minimum_repeat=2) -> dict:
        
        phrases = {}
        stopwords2 = stopwords.words('english')
        for text in texts:
            # Replace separators and punctuation with spaces
            text = re.sub(r'[.!?,:;/\-\s]', ' ', text)
            # Remove extraneous chars
            text = re.sub(r'[\\|@#$&~%\(\)*\"]', '', text)

            words = text.split(' ')
            # Remove stop words and empty strings
            words = [w for w in words if len(w) and w.lower() not in stopwords2]
            resultWords = []
            for item in words:
                if item not in resultWords:
                    resultWords.append(item)
            length = len(resultWords)
            # Look at phrases no longer than maximum_length words long
            size = length if length <= maximum_length else maximum_length
            while size > 0:
                pos = 0
                # Walk over all sets of words
                while pos + size <= length:
                    phrase = resultWords[pos:pos+size]
                    phrase = tuple(w.lower() for w in phrase)
                    if phrase in phrases:
                        phrases[phrase] += 1
                    else:
                        phrases[phrase] = 1
                    pos += 1
                size -= 1
        phrases = {k: v for k, v in phrases.items() if v >= minimum_repeat}
        longest_phrases = {}
        keys = list(phrases.keys())
        #print(keys)
        keys.sort(key=len, reverse=True)
        for phrase in keys:
            found = False
            for l_phrase in longest_phrases:
                # If the entire phrase is found in a longer tuple...
                intersection = set(l_phrase).intersection(phrase)
                if len(intersection) == len(phrase):
                    # ... and their frequency overlaps by 75% or more, we'll drop it
                    difference = (phrases[phrase] - longest_phrases[l_phrase]) / longest_phrases[l_phrase]
                    if difference < 0.25:
                        found = True
                        break
            if not found:
                longest_phrases[phrase] = phrases[phrase]

        return longest_phrases
    def countOverlappings(self, similitudeMatrix, minSimilitude):
        contSolapamientos=0
        newLabels=[]
        for i in range(len(self.data.toarray())):
            docs=[]
            cont1=0
            for j in range(self.numClusters):
                if(similitudeMatrix[i][j]>=minSimilitude):
                    docs.append(j)
                    cont1+=1
            newLabels.append(docs)
            if cont1>1:
                contSolapamientos+=1
        return newLabels,contSolapamientos 
    
    def countOverlappingsxGroup(self, similitudeMatrix, minSimilitudeGroup):
        contSolapamientos=0
        newLabels=[]
        for i in range(len(self.data.toarray())):
            clusters=[]
            cont1=0
            for j in range(self.numClusters):
                if(similitudeMatrix[j][i]>=minSimilitudeGroup[j]):
                    clusters.append(j)
                    cont1+=1
            if(len(clusters)==0):
                listpertenencia=[similitudeMatrix[cluster][i] for cluster in range(self.numClusters)]
                index=np.where(listpertenencia==listpertenencia.max())[0][0]
                clusters.append(index)
            newLabels.append(clusters)
            if cont1>1:
                contSolapamientos+=1
        return newLabels,contSolapamientos 
    def countOverlappingsxGroupOthers(self, distancesMatrix,labels):
        topes=self.overlappingOtherAlgorithms(distancesMatrix,labels)
        contSolapamientos=0
        newLabels=[]
        for i in range(len(self.data.toarray())):
            cluster=labels[i]
            clusters=[]
            clusters.append(cluster)
            if(distancesMatrix[cluster][i]>topes[cluster][1]):
                for numCluster in range(self.numClusters):
                    if(cluster!=numCluster):
                        #if(distancesMatrix[numCluster][i]>=topes[numCluster][0] and distancesMatrix[numCluster][i]<=topes[numCluster][1]):
                        if(distancesMatrix[numCluster][i]<topes[numCluster][1]):  
                            clusters.append(numCluster)
            newLabels.append(clusters)
            if(len(clusters)>1):
                contSolapamientos+=1
        return newLabels,contSolapamientos 
    
    def createDistancesMatrix(self, centroids, x2,labels):
        distancesMatrix=np.full(shape=(self.numClusters,len(self.data.toarray())),dtype=float, fill_value=10)
        for i in range(self.numClusters):
            numss=[x for x in range(len(self.data.toarray()))]
            numss=np.array(numss)
            indices = numss[labels == i]
            for index in indices:
                for j in range(self.numClusters):
                    distance=cosine_distances(x2[index],centroids[j])
                    distancesMatrix[j][index]=distance     
        return distancesMatrix
    
    def similitudeKMeans(self,distanceMatrix):
        minSimilitudeGroups=[]
        for i in range(distanceMatrix.shape[1]):
            min=1
            for j in range(distanceMatrix.shape[0]):
                if(distanceMatrix[j][i]<min):
                    min=distanceMatrix[j][i]
            minSimilitudeGroups.append(min)
        return minSimilitudeGroups
    def overlappingFuzzyCmeans(self,affinityMatrix):
        # calcular el rango intercuartílico 
        topes=[]
        for i in range(self.numClusters):
            listDocs=[j for j in affinityMatrix[i]]
            q34, q14 = np.percentile(listDocs, [75, 25])
            listDocs.sort(reverse=True)        
            q3, q1 = np.percentile(listDocs, [75, 25])
            quartileSubstraction =q1-(1.5*(q3 - q1))
            topes.append(abs(quartileSubstraction))         
        return topes
    def overlappingOtherAlgorithms(self,distanceMatrix, labels):
        # calcular el rango intercuartílico 
        topes=[]
        for i in range(self.numClusters):
            listDocs=[j for j in distanceMatrix[i][labels==i]]
            
            q34, q14 = np.percentile(listDocs, [75, 25])
            listDocs.sort(reverse=True)        
            q3, q1, media= np.percentile(listDocs, [75, 25,50])
            IQR=q3 - q1
            topeInferior =q1-(1.5*IQR)
            topeSuperior=q3+(1.5*IQR)
            topes.append((topeInferior,topeSuperior,media))         
        return topes
    def etiquetar_clusters(self,documentos, clusters, listDocsGroups):
        etiquetas = {}
        for i, cluster in enumerate(clusters):
            # Crear una lista de frases para cada cluster
            frases = ("").join(listDocsGroups[i]) 
            # Encontrar la frase más común en el cluster
            frase_comun = Counter(frases).most_common()[0][0]
            # Buscar la subcadena más larga en común
            for j in range(len(frase_comun), 0, -1):
                for k in range(len(frase_comun) - j + 1):
                    subcadena = frase_comun[k:k+j]
                    if all(subcadena in f for f in frases):
                        etiquetas[i] = subcadena
                        break
                else:
                    continue
                break
        return etiquetas
    def imprimirDocs(self,labels,docsEspanol):
        i=0
        for label in np.unique(labels):
            group=[]
            print(" \n Cluster "+ str(i)+": ")
            for j in range(len(labels)):
                if(labels[j]==label):
                    print(" \n Doc "+str(j)+": ")
                    print(docsEspanol[j])
            print("")
            i+=1
    def normalize(self,value, min_val, max_val, new_min, new_max):
        """
            Normaliza un valor en un rango específico.
            
            Parámetros:
            value (float o int): El valor a normalizar.
            min_val (float o int): El valor mínimo del rango original.
            max_val (float o int): El valor máximo del rango original.
            new_min (float o int): El valor mínimo del rango objetivo.
            new_max (float o int): El valor máximo del rango objetivo.
            
            Retorna:
            float: El valor normalizado en el rango objetivo.
        """
        normalized_value = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        return normalized_value

    def executeSTC(self,abstractsList,numeroClusters=None):
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters

        n_components=150 if self.data.shape[1]>=150 else self.data.shape[1]
        lsa = make_pipeline(TruncatedSVD(n_components=n_components))
        X_lsa = lsa.fit_transform(self.data.toarray())
        labelsOverlapped=[]
        for i in range(len(abstractsList)):
            labelsOverlapped.append([])
    
        documents=[]
        for abs in abstractsList:
            documents.append({"title":abs})
        body={
                "language": "English",
                "algorithm": "STC",
                "documents": documents,
                "parameters": {
                            "maxClusters": self.numClusters
                        }
        }
        url="http://localhost:8080/service/cluster"
        rta=requests.post(url,json=body)
            
        rta=json.loads(rta.content)
        
        countClusters=0
        if("clusters" in rta.keys()):
            indexCluster=0
            
            for cluster in rta["clusters"]:
                if("documents" in cluster.keys()):
                    
                    for doc in cluster["documents"]:
                       labelsOverlapped[doc].append(indexCluster)
                    
                    indexCluster=indexCluster+1
                    countClusters=countClusters+1
        band=False
        for index in range(len(labelsOverlapped)):
            if(len(labelsOverlapped[index])==0):
                labelsOverlapped[index].append(len(rta["clusters"]))
                if(band==False):
                    countClusters=countClusters+1
                    band=True

        self.numClusters=countClusters
        centroids = []
        for i in range(countClusters):
            vectorCondicion=[True if i in labelsOverlapped[j] else False for j in range(len(abstractsList)) ]
            articles=X_lsa[vectorCondicion]
            centroid = articles.mean(axis=0)
            centroids.append(centroid)
        return labelsOverlapped,X_lsa,centroids
    
    def executeLingo(self,abstractsList,numeroClusters=None):
        if(numeroClusters==None):
            self.numClusters=self.calculateNumClusters()
        else:
            self.numClusters=numeroClusters

        n_components=150 if self.data.shape[1]>=150 else self.data.shape[1]
        lsa = make_pipeline(TruncatedSVD(n_components=n_components))
        X_lsa = lsa.fit_transform(self.data.toarray())
        labelsOverlapped=[]
        for i in range(len(abstractsList)):
            labelsOverlapped.append([])
    
        documents=[]
        for abs in abstractsList:
            documents.append({"title":abs})
        body={
                "language": "English",
                "algorithm": "Lingo",
                "documents": documents,
                "parameters": {
                    "desiredClusterCount":  self.numClusters
                }
             
        }
        url="http://localhost:8080/service/cluster"
        rta=requests.post(url,json=body)
            
        rta=json.loads(rta.content)
       
        countClusters=0
        if("clusters" in rta.keys()):
            indexCluster=0
            for cluster in rta["clusters"]:
                if("documents" in cluster.keys()):
                    for doc in cluster["documents"]:
                       labelsOverlapped[doc].append(indexCluster)
                    indexCluster=indexCluster+1
                    countClusters=countClusters+1
        band=False
        for index in range(len(labelsOverlapped)):
            if(len(labelsOverlapped[index])==0):
                labelsOverlapped[index].append(len(rta["clusters"]))
                if(band==False):
                    countClusters=countClusters+1
                    band=True

        self.numClusters=countClusters
        centroids = []
        for i in range(countClusters):
            vectorCondicion=[True if i in labelsOverlapped[j] else False for j in range(len(abstractsList)) ]
            articles=X_lsa[vectorCondicion]
            centroid = articles.mean(axis=0)
            centroids.append(centroid)
        return labelsOverlapped,X_lsa,centroids
  
    

