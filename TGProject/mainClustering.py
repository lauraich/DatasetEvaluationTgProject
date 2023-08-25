import numpy as np
from DataSetProcessing import DataSetProcessing
from DocumentClustering import DocumentClustering
from textProcessing import textProcessing
from pruebaEvaluacion import Principal
from pruebaEvaluacionOverlapping import PrincipalOverlapping
import pandas as pd


dt=DataSetProcessing()

docs,y_true,numeroClases,y_true_ns=dt.getDataSetTopicModeling()

print("num clases",numeroClases)
objTXTProcesing=textProcessing()
X = objTXTProcesing.tfidf(docs)
mpd=pd.DataFrame(X)
mpd.to_csv("matrizTfIDFTopicModeling.csv")
print("Filas: ",X.shape[0])
print("Columnas: ",X.shape[1])
dc=DocumentClustering(data=X,maxClusters=numeroClases)
kmeans_avg=np.zeros(shape=15,dtype=float)
spectral_avg=np.zeros(shape=15,dtype=float)
fuzzy_avg=np.zeros(shape=15,dtype=float)
stc_avg=np.zeros(shape=15,dtype=float)
lingo_avg=np.zeros(shape=15,dtype=float)

numEjecuciones = 31
#Ejecución de 31 ejecuciones de los algoritmos kmeans, spectral y fuzzy

for iteracion in range(numEjecuciones):
    print("Iteracion: ",iteracion)
    numClusters=dc.calculateNumClusters()
    np.random.seed(iteracion)
    
    #------------para KMEANS---------------------
    X_lsa_kmeans=dc.clusteringKMeans(numeroClusters=numClusters)
    labelsk=dc.kmeansClusterResult.labels_
    distanceMatrixK=dc.createDistancesMatrix(dc.kmeansClusterResult.cluster_centers_,X_lsa_kmeans,labels=labelsk)
    newlabels_k, co=dc.countOverlappingsxGroupOthers(distanceMatrixK,labelsk)
    principalOverlappingK = PrincipalOverlapping(data=X.toarray(),labels_true=y_true,labels_pred=newlabels_k,cantidadDeClases=numeroClases, cantidadDeClusters=numClusters)
    ratiok=dc.SSE(dc.kmeansClusterResult.cluster_centers_, X_lsa_kmeans, labelsk)
    #PrecisionPonderadaK, RecuerdoPonderadoK, MedidaFPonderadaK, NiccK, PiccK,WtprK,WfprK, WAccuracyK, WnpvK,WfdrK=principalOverlappingK.getMetrics()
    metricsK=principalOverlappingK.getMetrics()
    kmeans_avg[0]+=ratiok
    for i in range(1,15):
        kmeans_avg[i]+=float(metricsK[i-1])
    #-------------- Para Spectral
    X_lsaS,lsaS=dc.clusteringSpectral(numeroClusters=numClusters)
    labelsS=dc.spectralClusterResult.labels_ 
    # Calculate the centroid of each cluster
    centroids = []
    for i in range(numClusters):
        centroid = X_lsaS[labelsS == i].mean(axis=0)
        centroids.append(centroid)
    centroids_for_labels = [] 
    ratios=dc.SSE(centroids, X_lsaS, labelsS)
    distancesMatrixS=dc.createDistancesMatrix(centroids,X_lsaS,labels=labelsS)
    newlabelsS, co=dc.countOverlappingsxGroupOthers(distancesMatrixS, labelsS)
    principalS = PrincipalOverlapping(data=X.toarray(),labels_true=y_true,labels_pred=newlabelsS,cantidadDeClases=numeroClases, cantidadDeClusters=numClusters)
    #PrecisionPonderadaS, RecuerdoPonderadoS, MedidaFPonderadaS, NiccS, PiccS,WtprS,WfprS, WAccuracyS, WnpvS,WfdrS=principalS.getMetrics()
    metricsS=principalS.getMetrics()
    spectral_avg[0]+=ratios
    for i in range(1,15):
        spectral_avg[i]+=float(metricsS[i-1])
    #-------------PARA FUZZY C MEANS
    labelsF, cantSolapamientosF,matrizMembresia=dc.clusteringFuzzyCMeansV2(len(docs),numeroClusters=numClusters) 
    label_pred=[]
    for lbl in labelsF:
        label_pred.append(lbl[0])
    centroidsF = []
    for i in range(dc.numClusters):
        vectorCondicion=[True if i in labelsF[j] else False for j in range(len(labelsF)) ]
        articles=X.toarray()[vectorCondicion]
        index2=0
        for indexArticle in np.where(np.array(vectorCondicion)==True)[0]:
            
            articles[index2]=articles[index2]*matrizMembresia[i][indexArticle]
            index2+=1
        centroid = articles.mean(axis=0)
        centroidsF.append(centroid)
    ratioF=dc.SSE(np.array(centroidsF), X.toarray(), label_pred)
    #labelPrueba=[lbl[i] for i in range(len(lbl)) for lbl in labelsF]
    #print("labels", np.unique(labelPrueba))
    principalF = PrincipalOverlapping(data=X.toarray(),labels_true=y_true,labels_pred=labelsF,cantidadDeClases=numeroClases, cantidadDeClusters=numClusters)
    #PrecisionPonderadaF, RecuerdoPonderadoF, MedidaFPonderadaF, NiccF, PiccF,WtprF,WfprF, WAccuracyF, WnpvF,WfdrF=principalF.getMetrics()
    metricsF=principalF.getMetrics()
    fuzzy_avg[0]+=ratioF
    for i in range(1,15):
        fuzzy_avg[i]+=float(metricsF[i-1])
    #-------------PARA STC
    labelsStc,X_lsaStc,centroidsStc= dc.executeSTC(docs,numeroClusters=numClusters)
    label_predStc=[]
    for lbl in labelsStc:
        label_predStc.append(lbl[0])
    ratioStc=dc.SSE(np.array(centroidsStc), X_lsaStc, label_predStc)
    #Evaluación
    principalStc = PrincipalOverlapping(data=X.toarray(),labels_true=y_true,labels_pred=labelsStc,cantidadDeClases=numeroClases, cantidadDeClusters=dc.numClusters)
    print("total",np.sum(principalStc.TotalesPorCluster))
    metricsStc=principalStc.getMetrics()
    stc_avg[0]+=ratioStc
    for i in range(1,15):
        stc_avg[i]+=float(metricsStc[i-1])
    #-------------PARA Lingo
    labelsLingo,X_lsaLingo,centroidsLingo= dc.executeLingo(docs,numeroClusters=numClusters)
    label_predLingo=[]
    for lbl in labelsLingo:
        label_predLingo.append(lbl[0])
    ratioLingo=dc.SSE(np.array(centroidsLingo), X_lsaLingo, label_predLingo)
    #Evaluación
    principalLingo = PrincipalOverlapping(data=X.toarray(),labels_true=y_true,labels_pred=labelsLingo,cantidadDeClases=numeroClases, cantidadDeClusters=dc.numClusters)
    print("total",np.sum(principalLingo.TotalesPorCluster))
    metricsLingo=principalLingo.getMetrics()
    lingo_avg[0]+=ratioLingo
    for i in range(1,15):
        lingo_avg[i]+=float(metricsLingo[i-1])
kmeans_avg=kmeans_avg/numEjecuciones
spectral_avg=spectral_avg/numEjecuciones
fuzzy_avg=fuzzy_avg/numEjecuciones
stc_avg=stc_avg/numEjecuciones
lingo_avg=lingo_avg/numEjecuciones

reporte = "-------REPORTE KMEANS-------------------\n"
reporte += "Weighted Avg. Precision = " + str(kmeans_avg[1]) + "\n"
reporte += "Weighted Avg. Recuerdo  = " + str(kmeans_avg[2]) + "\n"
reporte += "Weighted Avg. F-measure = " + str(kmeans_avg[3]) + "\n"
reporte += "Instancias correctamente agrupadas = " + str(kmeans_avg[4]) + "\n"
reporte += "% Instancias correctamente agrupadas = " + str(kmeans_avg[5]) + "\n"
reporte += "Rata ponderada de verdaderos positivos =" + str(kmeans_avg[6]) + "\n"
reporte += "Rata ponderada de falsos positivos =" + str(kmeans_avg[7]) + "\n"
reporte += "Exactitud promedio ponderada (Accuracy) =" + str(kmeans_avg[8]) + "\n"
reporte += "Negative predictive value (NPV) =" + str(kmeans_avg[9]) + "\n"
reporte += "Rata ponderada de falsos descubrimientos (FDR) =" + str(kmeans_avg[10]) + "\n"
reporte += "SSE Avg= "+str(kmeans_avg[0])+"\n"
reporte += "Verdaderos Positivos= "+str(kmeans_avg[11])+ "\n"
reporte += "Verdaderos Negativos= "+str(kmeans_avg[12])+ "\n"
reporte += "Falsos Positivos= "+str(kmeans_avg[13])+ "\n"
reporte += "Falsos Negativos= "+str(kmeans_avg[14])+ "\n"
reporte += "Precision tp/(tp+fp)= "+str(kmeans_avg[11]/(kmeans_avg[11]+kmeans_avg[13]))+" \n"

reporteS = "-------REPORTE SPECTRAL-------------------\n"
reporteS += "Weighted Avg. Precision = " + str(spectral_avg[1]) + "\n"
reporteS += "Weighted Avg. Recuerdo  = " + str(spectral_avg[2]) + "\n"
reporteS += "Weighted Avg. F-measure = " + str(spectral_avg[3]) + "\n"
reporteS += "Instancias correctamente agrupadas = " + str(spectral_avg[4]) + "\n"
reporteS += "% Instancias correctamente agrupadas = " + str(spectral_avg[5]) + "\n"
reporteS += "Rata ponderada de verdaderos positivos =" + str(spectral_avg[6]) + "\n"
reporteS += "Rata ponderada de falsos positivos =" + str(spectral_avg[7]) + "\n"
reporteS += "Exactitud promedio ponderada (Accuracy) =" + str(spectral_avg[8]) + "\n"
reporteS += "Negative predictive value (NPV) =" + str(spectral_avg[9]) + "\n"
reporteS += "Rata ponderada de falsos descubrimientos (FDR) =" + str(spectral_avg[10]) + "\n"
reporteS += "SSE Avg= "+str(spectral_avg[0])+"\n"
reporteS += "Verdaderos Positivos= "+str(spectral_avg[11])+ "\n"
reporteS += "Verdaderos Negativos= "+str(spectral_avg[12])+ "\n"
reporteS += "Falsos Positivos= "+str(spectral_avg[13])+ "\n"
reporteS += "Falsos Negativos= "+str(spectral_avg[14])+ "\n"
reporteS += "Precision tp/(tp+fp)= "+str(spectral_avg[11]/(spectral_avg[11]+spectral_avg[13]))+" \n"
       
reporteF = "-------REPORTE FUZZY-------------------\n"
reporteF += "Weighted Avg. Precision = " + str(fuzzy_avg[1]) + "\n"
reporteF += "Weighted Avg. Recuerdo  = " + str(fuzzy_avg[2]) + "\n"
reporteF += "Weighted Avg. F-measure = " + str(fuzzy_avg[3]) + "\n"
reporteF += "Instancias correctamente agrupadas = " + str(fuzzy_avg[4]) + "\n"
reporteF += "% Instancias correctamente agrupadas = " + str(fuzzy_avg[5]) + "\n"
reporteF += "Rata ponderada de verdaderos positivos =" + str(fuzzy_avg[6]) + "\n"
reporteF += "Rata ponderada de falsos positivos =" + str(fuzzy_avg[7]) + "\n"
reporteF += "Exactitud promedio ponderada (Accuracy) =" + str(fuzzy_avg[8]) + "\n"
reporteF += "Negative predictive value (NPV) =" + str(fuzzy_avg[9]) + "\n"
reporteF += "Rata ponderada de falsos descubrimientos (FDR) =" + str(fuzzy_avg[10]) + "\n"
reporteF += "SSE Avg= "+str(fuzzy_avg[0])+"\n"
reporteF += "Verdaderos Positivos= "+str(fuzzy_avg[11])+ "\n"
reporteF += "Verdaderos Negativos= "+str(fuzzy_avg[12])+ "\n"
reporteF += "Falsos Positivos= "+str(fuzzy_avg[13])+ "\n"
reporteF += "Falsos Negativos= "+str(fuzzy_avg[14])+ "\n"
reporteF += "Precision tp/(tp+fp)= "+str(fuzzy_avg[11]/(fuzzy_avg[11]+fuzzy_avg[13]))+" \n"

reporteStc = "-------REPORTE STC-------------------\n"
reporteStc += "Weighted Avg. Precision = " + str(stc_avg[1]) + "\n"
reporteStc += "Weighted Avg. Recuerdo  = " + str(stc_avg[2]) + "\n"
reporteStc += "Weighted Avg. F-measure = " + str(stc_avg[3]) + "\n"
reporteStc += "Instancias correctamente agrupadas = " + str(stc_avg[4]) + "\n"
reporteStc += "% Instancias correctamente agrupadas = " + str(stc_avg[5]) + "\n"
reporteStc += "Rata ponderada de verdaderos positivos =" + str(stc_avg[6]) + "\n"
reporteStc += "Rata ponderada de falsos positivos =" + str(stc_avg[7]) + "\n"
reporteStc += "Exactitud promedio ponderada (Accuracy) =" + str(stc_avg[8]) + "\n"
reporteStc += "Negative predictive value (NPV) =" + str(stc_avg[9]) + "\n"
reporteStc += "Rata ponderada de falsos descubrimientos (FDR) =" + str(stc_avg[10]) + "\n"
reporteStc += "SSE Avg= "+str(stc_avg[0])+"\n"
reporteStc += "Verdaderos Positivos= "+str(stc_avg[11])+ "\n"
reporteStc += "Verdaderos Negativos= "+str(stc_avg[12])+ "\n"
reporteStc += "Falsos Positivos= "+str(stc_avg[13])+ "\n"
reporteStc += "Falsos Negativos= "+str(stc_avg[14])+ "\n"
reporteStc += "Precision tp/(tp+fp)= "+str(stc_avg[11]/(stc_avg[11]+stc_avg[13]))+" \n"

reporteLingo = "-------REPORTE LINGO-------------------\n"
reporteLingo += "Weighted Avg. Precision = " + str(lingo_avg[1]) + "\n"
reporteLingo += "Weighted Avg. Recuerdo  = " + str(lingo_avg[2]) + "\n"
reporteLingo += "Weighted Avg. F-measure = " + str(lingo_avg[3]) + "\n"
reporteLingo += "Instancias correctamente agrupadas = " + str(lingo_avg[4]) + "\n"
reporteLingo += "% Instancias correctamente agrupadas = " + str(lingo_avg[5]) + "\n"
reporteLingo += "Rata ponderada de verdaderos positivos =" + str(lingo_avg[6]) + "\n"
reporteLingo += "Rata ponderada de falsos positivos =" + str(lingo_avg[7]) + "\n"
reporteLingo += "Exactitud promedio ponderada (Accuracy) =" + str(lingo_avg[8]) + "\n"
reporteLingo += "Negative predictive value (NPV) =" + str(lingo_avg[9]) + "\n"
reporteLingo += "Rata ponderada de falsos descubrimientos (FDR) =" + str(lingo_avg[10]) + "\n"
reporteLingo += "SSE Avg= "+str(lingo_avg[0])+"\n"
reporteLingo += "Verdaderos Positivos= "+str(lingo_avg[11])+ "\n"
reporteLingo += "Verdaderos Negativos= "+str(lingo_avg[12])+ "\n"
reporteLingo += "Falsos Positivos= "+str(lingo_avg[13])+ "\n"
reporteLingo += "Falsos Negativos= "+str(lingo_avg[14])+ "\n"
reporteLingo += "Precision tp/(tp+fp)= "+str(lingo_avg[11]/(lingo_avg[11]+lingo_avg[13]))+" \n"

import datetime
fecha=datetime.datetime.now()
#name="reporte"+str(fecha)+".txt"
name="reporte"+".txt"
archivo=open(name,'w')
archivo.write(reporte)
archivo.write(reporteS)
archivo.write(reporteF)
archivo.write(reporteStc)
archivo.write(reporteLingo)
archivo.close()
print(reporte)
print(reporteS)
print(reporteF)
print(reporteStc)
print(reporteLingo)