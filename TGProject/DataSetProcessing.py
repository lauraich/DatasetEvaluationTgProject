import json
import pandas as pd
import numpy as np
from textProcessing import textProcessing

class DataSetProcessing:
    def __init__(self) -> None:
        pass
    
    def getDataSetAAAI14(self):
        link="https://archive.ics.uci.edu/ml/datasets/AAAI+2014+Accepted+Papers"
        meta_data = pd.read_csv("DataSets/AAAI-14.csv")
        docs=[]
        objTXTProcesing=textProcessing()
        sad=meta_data['abstract'].tolist()
        title=meta_data['title'].tolist()
        keywords=meta_data['keywords'].tolist()
        text =[]
        for item in range(len(sad)):
            text.append(sad[item]+" "+title[item]+" "+keywords[item])
        topics=meta_data['groups'].tolist()
        for x in range(len(topics)):
            if str(topics[x]) == 'nan':
                text.pop(x)
        docs=objTXTProcesing.preProcesingListData(text)
        topics=[str(x).split("\n") for x in topics if str(x) != 'nan']
        
        newTopics=[]
        for topic in topics:
            for x in topic:
                newTopics.append(x)
        y=np.unique(newTopics)
        """ 
        for x in y:
            print(x) """
        dictt={}
        for item in range(len(y)):
            dictt[y[item]]=item
        topicsn=[]
        for x in topics:
            topic=[dictt[t] for t in x]
            topicsn.append(topic)
        #y_true=meta_data['Y1']
        y_true=topicsn
        numeroClases=len(y)
        topicNoSolap=[]
        for topic in y_true:            
            topicNoSolap.append(topic[0])
        dictNoSolap={}
        uniqueTopicNoSolap=np.unique(topicNoSolap)
        for item in range(len(uniqueTopicNoSolap)):
            dictNoSolap[uniqueTopicNoSolap[item]]=item
        topicNS=[]
        for x in topicNoSolap:
            topicNS.append(dictNoSolap[x])
        return docs, y_true,numeroClases,topicNS
    def getDataSetAAAI13(self):
        link="https://archive.ics.uci.edu/ml/datasets/AAAI+2014+Accepted+Papers"
        meta_data = pd.read_csv("DataSets/AAAI-13.csv")
        docs=[]
        objTXTProcesing=textProcessing()
        sad=meta_data['Abstract'].tolist()
        title=meta_data['Title'].tolist()
        keywords=meta_data['Keywords'].tolist()
        text =[]
        for item in range(len(sad)):
            if(str(sad[item])=='NaN'):
                sad[item]=""
            if(str(title[item])=='NaN'):
                title[item]=""
            if(str(keywords[item])=='NaN'):
                keywords[item]=""
            text.append(sad[item]+" "+title[item]+" "+keywords[item])
        topics=meta_data['Topics'].tolist()
        for x in range(len(topics)):
            if str(topics[x]) == 'nan':
                text.pop(x)
        docs=objTXTProcesing.preProcesingListData(text)
        topics=[str(x).split("\n") for x in topics if str(x) != 'nan']
        newTopics=[]
        for topic in topics:
            for x in topic:
                newTopics.append(x)
        y=np.unique(newTopics)
        
        dictt={}
        for item in range(len(y)):
            dictt[y[item]]=item
        topicsn=[]
        for x in topics:
            topic=[dictt[t] for t in x]
            topicsn.append(topic)
        y_true=topicsn
        numeroClases=len(y)
        topicNoSolap=[]
        for topic in y_true:            
            topicNoSolap.append(topic[0])
        dictNoSolap={}
        uniqueTopicNoSolap=np.unique(topicNoSolap)
        for item in range(len(uniqueTopicNoSolap)):
            dictNoSolap[uniqueTopicNoSolap[item]]=item
        topicNS=[]
        for x in topicNoSolap:
            topicNS.append(dictNoSolap[x])
        return docs, y_true,numeroClases,topicNS
    """     def getDataSet3Reducido(self):
        meta_data = pd.read_csv("DataSets/3Reducido.csv", sep=';')
        docs=[]
        objTXTProcesing=textProcessing()
        sad=meta_data['abstract'].tolist()
        topics=meta_data['groups'].tolist()
        for x in range(len(topics)):
            if str(topics[x]) == 'nan':
                sad.pop(x)
        docs=objTXTProcesing.preProcesingListData(sad)
        topics=[str(x).split("\n") for x in topics if str(x) != 'nan']
        newTopics=[]
        for topic in topics:
            for x in topic:
                newTopics.append(x)
        y=np.unique(newTopics)
       
        #for x in y:
        #    print(x) 
        dictt={}
        for item in range(len(y)):
            dictt[y[item]]=item
        topicsn=[]
        for x in topics:
            topic=[dictt[t] for t in x]
            topicsn.append(topic)
        #y_true=meta_data['Y1']
        y_true=topicsn
        numeroClases=len(y)
        topicNoSolap=[]
        for topic in y_true:            
            topicNoSolap.append(topic[0])
        dictNoSolap={}
        uniqueTopicNoSolap=np.unique(topicNoSolap)
        for item in range(len(uniqueTopicNoSolap)):
            dictNoSolap[uniqueTopicNoSolap[item]]=item
        topicNS=[]
        for x in topicNoSolap:
            topicNS.append(dictNoSolap[x])
        return docs, y_true,numeroClases,topicNS 
    """
    def getDataSet2(self):
        #El nombre del dataset es WOS46985
        meta_data = pd.read_csv("DataSets/WOS46985.csv")
        docs=[]
        objTXTProcesing=textProcessing()
        sad=meta_data['Abstract'].tolist()
        docs=objTXTProcesing.preProcesingListData(sad)
        y_true=meta_data['Y1'].tolist()
        y=np.unique(y_true)
        numeroClases=len(y)
        y_true_lista=[[x] for x in y_true]
        return docs, y_true_lista,numeroClases, y_true
    def getDataSetTopicModeling(self):
        meta_data = pd.read_csv("DataSets/topicModeling.csv")
        docs=[]
        objTXTProcesing=textProcessing()
        abstract=meta_data['ABSTRACT'].tolist()
        docs=objTXTProcesing.preProcesingListData(abstract)
        ComputerScience=meta_data['Computer Science'].tolist()
        Mathematics=meta_data['Mathematics'].tolist()
        Physics=meta_data['Physics'].tolist()
        Statistics=meta_data['Statistics'].tolist()
        AnalysisofPDEs=meta_data['Analysis of PDEs'].tolist()
        ArtificialIntelligence=meta_data['Artificial Intelligence'].tolist()
        AstrophysicsofGalaxies=meta_data['Astrophysics of Galaxies'].tolist()
        ComputationandLanguage=meta_data['Computation and Language'].tolist()
        ComputerVisionandPatternRecognition=meta_data['Computer Vision and Pattern Recognition'].tolist()
        CosmologyandNongalacticAstrophysics=meta_data['Cosmology and Nongalactic Astrophysics'].tolist()
        DataStructuresandAlgorithms=meta_data['Data Structures and Algorithms'].tolist()
        DifferentialGeometry=meta_data['Differential Geometry'].tolist()
        EarthandPlanetaryAstrophysics=meta_data['Earth and Planetary Astrophysics'].tolist()
        FluidDynamics=meta_data['Fluid Dynamics'].tolist()
        InformationTheory=meta_data['Information Theory'].tolist()
        InstrumentationandMethodsforAstrophysics=meta_data['Instrumentation and Methods for Astrophysics'].tolist()
        MachineLearning=meta_data['Machine Learning'].tolist()
        MaterialsScience=meta_data['Materials Science'].tolist()
        Methodology=meta_data['Methodology'].tolist()
        NumberTheory=meta_data['Number Theory'].tolist()
        OptimizationandControl=meta_data['Optimization and Control'].tolist()
        RepresentationTheory=meta_data['Representation Theory'].tolist()
        Robotics=meta_data['Robotics'].tolist()
        SocialandInformationNetworks=meta_data['Social and Information Networks'].tolist()
        StatisticsTheory=meta_data['Statistics Theory'].tolist()
        StronglyCorrelatedElectrons=meta_data['Strongly Correlated Electrons'].tolist()
        Superconductivity=meta_data['Superconductivity'].tolist()
        SystemsandControl=meta_data['Systems and Control'].tolist()
        docs=[docs[i] for i in range(100)]
        topics=[]
        for index in range(len(docs)):
            lbl=[]
            if(ComputerScience[index]==1):
                lbl.append(0)
            if(Mathematics[index]==1):
                lbl.append(1)
            if(Physics[index]==1):
                lbl.append(2)
            if(Statistics[index]==1):
                lbl.append(3)
            if(AnalysisofPDEs[index]==1):
                lbl.append(4)
            if(ArtificialIntelligence[index]==1):
                lbl.append(5)
            if(AstrophysicsofGalaxies[index]==1):
                lbl.append(6)
            if(ComputationandLanguage[index]==1):
                lbl.append(7)
            if(ComputerVisionandPatternRecognition[index]==1):
                lbl.append(8)
            if(CosmologyandNongalacticAstrophysics[index]==1):
                lbl.append(9)
            if(DataStructuresandAlgorithms[index]==1):
                lbl.append(10)
            if(DifferentialGeometry[index]==1):
                lbl.append(11)
            if(EarthandPlanetaryAstrophysics[index]==1):
                lbl.append(12)
            if(FluidDynamics[index]==1):
                lbl.append(13)
            if(InformationTheory[index]==1):
                lbl.append(14)
            if(InstrumentationandMethodsforAstrophysics[index]==1):
                lbl.append(15)
            if(MachineLearning[index]==1):
                lbl.append(16)
            if(MaterialsScience[index]==1):
                lbl.append(17)
            if(Methodology[index]==1):
                lbl.append(18)
            if(NumberTheory[index]==1):
                lbl.append(19)
            if(OptimizationandControl[index]==1):
                lbl.append(20)
            if(RepresentationTheory[index]==1):
                lbl.append(21)
            if(Robotics[index]==1):
                lbl.append(22)
            if(SocialandInformationNetworks[index]==1):
                lbl.append(23)
            if(StatisticsTheory[index]==1):
                lbl.append(24)
            if(StronglyCorrelatedElectrons[index]==1):
                lbl.append(25)
            if(Superconductivity[index]==1):
                lbl.append(26)
            if(ComputerScience[index]==1):
                lbl.append(27)
            if(SystemsandControl[index]==1):
                lbl.append(28)
            topics.append(lbl)
        y_true_lista=topics
        y_true=[x[0] for x in topics]
        return docs, y_true_lista,29, y_true
    
    def getDataSetArxiv(self):
        lista=[]
        with open("DataSets/arxiv.json", 'r') as datafile:
            for line in datafile:
                lista.append(line)
        abstracts=[]
        y_true_lista=[]   
        y_true=[]     
        docs=[]
        docs=[json.loads(x) for x in lista[0:1000]]
        abstracts=[x['abstract'] +" "+x['title'] for x in docs]
        categorias=[]
        y_true_lista=[]
        for x in docs:
            y_true_lista.append(x['categories'].split(' '))
            y_true.append(x['categories'].split(' ')[0])
            for y in x['categories'].split(' '):
                categorias.append(y)
        categorias=np.unique(categorias)
        categorias=[x.split('.')[0] for x in categorias]
        categorias=np.unique(categorias)
        dictCategoriaToNumber={}
        numeroClases=len(categorias)
        contador=0
        for cat in categorias:
            dictCategoriaToNumber[cat]=contador
            contador+=1
        for i in range(len(y_true_lista)):
            for j in range(len(y_true_lista[i])):
                x=y_true_lista[i]
                x[j]=dictCategoriaToNumber[x[j].split('.')[0]]
            y_true_lista[i]=np.unique(y_true_lista[i])
        ###############################
        categorias=np.unique(y_true)
        categorias=[x.split('.')[0] for x in categorias]
        categorias=np.unique(categorias)
        dictCategoriaToNumber={}
        contador=0
        for cat in categorias:
            dictCategoriaToNumber[cat]=contador
            contador+=1
        for i in range(len(y_true)):
            y_true[i]=dictCategoriaToNumber[y_true[i].split('.')[0]]

        return abstracts, y_true_lista, numeroClases, y_true
    
"""  def getDataSetScopus(self):
        lista=[]
        with open("./DataSets/datosScopusR.json", 'r') as datafile:
            for line in datafile:
                lista.append(line)
        abstracts=[]
        y_true_lista=[]   
        y_true=[]     
        docs=[]
        docs=[json.loads(x) for x in lista]
        abstracts=[x['abstract'] for x in docs]
        categorias=[]
        y_true_lista=[]
        for x in docs:
            y_true_lista.append(x['topic'].split(','))
            y_true.append(x['topic'].split(',')[0])
            for y in x['topic'].split(','):
                categorias.append(y)
        categorias=np.unique(categorias)
        print("Len: ",len(categorias))
        
        
        dictCategoriaToNumber={}
        numeroClases=len(categorias)
        contador=0
        for cat in categorias:
            dictCategoriaToNumber[cat]=contador
            contador+=1
        for i in range(len(y_true_lista)):
            for j in range(len(y_true_lista[i])):
                x=y_true_lista[i]
                x[j]=dictCategoriaToNumber[x[j]]
            y_true_lista[i]=np.unique(y_true_lista[i])
        ###############################
        categorias=np.unique(y_true)
        dictCategoriaToNumber={}
        contador=0
        for cat in categorias:
            dictCategoriaToNumber[cat]=contador
            contador+=1
        for i in range(len(y_true)):
            y_true[i]=dictCategoriaToNumber[y_true[i]]

        return abstracts, y_true_lista, numeroClases, y_true
"""
    
