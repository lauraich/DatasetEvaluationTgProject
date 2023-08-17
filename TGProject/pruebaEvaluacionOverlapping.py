import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
import pandas as pd
class PrincipalOverlapping:
    def __init__(self, labels_true,labels_pred, cantidadDeClases, cantidadDeClusters, data):
        self.data=data
        self.labels_true=labels_true
        self.labels_pred=labels_pred
        lsa = make_pipeline(TruncatedSVD(n_components=100))
        self.X_lsa = lsa.fit_transform(self.data)
        matrizdatosdetallados = np.zeros(shape=(len(labels_true), 3), dtype=np.ndarray)
        for i in range(len(labels_true)):
            matrizdatosdetallados[i][0] = i
            matrizdatosdetallados[i][1] = labels_true[i]
            matrizdatosdetallados[i][2] = labels_pred[i]
        # Initialize the attributes
        self.Nicc = 0 # numero de instancias correctamente clasificadas
        self.Picc = 0 # Porcentaje de instancias correctamente clasificadas
        self.PrecisionPonderada = 0
        self.RecuerdoPonderado = 0
        self.MedidaFPonderada = 0
        self.Wtpr = 0 # Rata ponderada de verdaderos positivos = Recall = RecuerdoPonderado
        self.Wfpr = 0 # Rata ponderada de falsos positivos = Fall-Out
        self.WAccuracy = 0 # Exactitud promedio ponderada = Accuracy = Rand Index
        self.WSpecificity = 0 # Especificidad promedio ponderada ( 1- Wfpr) = Specificity - contrario a rata de falsos positivos
        self.Wnpv = 0 # Negative predictive Value ponderada = NPV
        self.Wfdr = 0 # Rata de Falsos descubrimientos ponderada = FDR

            # Initialize the attributes
        self.TotalClases = cantidadDeClases
        self.TotalClusters = cantidadDeClusters
        self.TotalDocumentos = len(matrizdatosdetallados)


         # Initialize the best and current arrays
        self.Best = np.full(shape=self.TotalClusters + 2, dtype=float,fill_value=-1)
        self.current = np.zeros(self.TotalClusters + 1)
        self.Best[self.TotalClusters] = float(self.TotalClases)
        self.Best[self.TotalClusters+1] = float('inf')

        # Map the classes and calculate the measures
        self.MapClasses2()


        # Convert the matrix to a numpy array for easier manipulation
        self.MatrixDeConfusion = np.zeros((self.TotalClusters+1, self.TotalClases+1), dtype=int)
        """   for i in range(self.TotalDocumentos):
            idDocumento, idClaseReal, idGrupoEncontrado = matrizdatosdetallados[i]
            for grupo in idGrupoEncontrado:
                for clase in idClaseReal:
                    self.MatrixDeConfusion[grupo][clase] += 1     
        """
        for i in range(self.TotalDocumentos):
            idDocumento, clasesReales, clustersPred = matrizdatosdetallados[i]
            
            clustersPredTraducidos=[]
            for k in range(len(clustersPred)):
                if self.Best[clustersPred[k]] == -1: continue
                clustersPredTraducidos.append(self.Best[clustersPred[k]])
            intersection=np.intersect1d(clasesReales,clustersPredTraducidos)
            #Se suma en la matriz de confusión los que están bien agrupados.
            #Lo que se suma es 1/(cantidad etiquetas predichas del documento)
            for clase in intersection:
                cluster=np.where(self.Best==clase)
                self.MatrixDeConfusion[int(cluster[0][0])][int(clase)] += 1
            ##Se sacan las etiquetas que no están en la intersección
            #es decir las erroneas
            wrongPred=[x for x in clustersPredTraducidos if x not in intersection]
            wrongReal=[x for x in clasesReales if x not in intersection]
            #marcar los errores
            for wcluster in wrongPred:
                cluster=np.where(self.Best==wcluster)
                for wclase in wrongReal:
                    self.MatrixDeConfusion[int(cluster[0][0])][int(wclase)] += 1
            #Identificar y marcar los huerfanos de cluster
            #Es decir clases que no tienen cluster [] [1]
            if(len(wrongPred)==0 and len(wrongReal)!=0):
                for wclase in wrongReal:
                    self.MatrixDeConfusion[self.TotalClusters][int(wclase)] += 1
            #Identificar los huerfanos de clases
            #Es decir cluster que no tienen clase [1] [ ]
            if(len(wrongPred)!=0 and len(wrongReal)==0):
                for wcluster in wrongPred:
                    cluster=np.where(self.Best==wcluster)
                    self.MatrixDeConfusion[int(cluster[0][0])][self.TotalClases] += 1
            """ for grupo in idGrupoEncontrado:
                incremento=1
                #cluster 3=clase 5
                
                
                if(int(self.Best[grupo]) in clasesReal):
                    self.MatrixDeConfusion[grupo][int(self.Best[grupo])] += 1/len(idGrupoEncontrado)
                    incremento= 
                else:
                    self.MatrixDeConfusion[grupo][self.TotalClases] += 1
            #traducimos las clases predichas
            traduccionGrupo=[ int(self.Best[x]) for x in idGrupoEncontrado]
            #Verficamos si hay alguna clase real huerfana
            for classReal in clasesReal:
                if(classReal not in traduccionGrupo):
                    self.MatrixDeConfusion[self.TotalClusters][classReal] += 1  """ 
                      
        #mc=pd.DataFrame(self.MatrixDeConfusion)
        #mc.to_csv("matrizconfusion.csv",sep=";")
        #a=[row[:-1] for row in self.MatrixDeConfusion[:-1]]
        # Calculate the totals by cluster and class
        self.TotalesPorCluster = np.sum(self.MatrixDeConfusion, axis=1)
        self.TotalesPorClase = np.sum(self.MatrixDeConfusion, axis=0)   
        
        """ current = np.zeros(self.TotalClusters + 1)
        self.MapClasses(0,current,0) """
        self.CalcularMedidas()
    def getBest(self):
        return self.Best
    def calcularCentroides(self):
        self.centroids_true = []
        for i in range(self.TotalClases):
            vectorCondicion=[False for x in range(self.TotalDocumentos)]
            index=0
            for lbl in self.labels_true:
                if i in lbl:
                    vectorCondicion[index]=True
                index+=1
            centroid = self.X_lsa[vectorCondicion].mean(axis=0)
            self.centroids_true.append(centroid)
        self.centroids_pred = []
        for i in range(self.TotalClusters):
            vectorCondicion=[False for x in range(self.TotalDocumentos)]
            index=0
            for lbl in self.labels_pred:
                if i in lbl:
                    vectorCondicion[index]=True
                index+=1
            centroid = self.X_lsa[vectorCondicion].mean(axis=0)
            self.centroids_pred.append(centroid)
    def MapClasses2(self):
        #centroid_similitudes=self.averageLinkage()
        #centroid_similitudes=self.completeLinkage()
        #centroid_similitudes=self.simpleLinkage()
        centroid_similitudes=self.centroidLinkage()
        band=True
        while(len(centroid_similitudes)>0):
            centroid_similitudes=sorted(centroid_similitudes, key=lambda x: x[2], reverse=True)
            if(band):
                #pand=pd.DataFrame(centroid_similitudes)
                #pand.to_csv("centroid_similitudes.csv")
                band=False
            claseReal, clasePred, similitude=centroid_similitudes[0]
            self.Best[clasePred]=claseReal
            centroid_similitudes=[x for x in centroid_similitudes if x[0]!=claseReal and x[1]!=clasePred]
        """ 
        newclasePred = np.copy(self.labels_pred)
        for clasePredicha in range(self.Best):
            if(clasePredicha!=self.Best[clasePred]):
                for doc in range(len(self.labels_pred)):
                    for labels in range(len(doc)):
                        if(self.labels_pred[doc][labels]==clasePredicha):
                            newclasePred[doc][labels]=self.Best[clasePredicha]
        """

    def centroidLinkage(self):
        self.calcularCentroides()
        centroid_similitudes=[]
        for claseReal in range(self.TotalClases):
            for clasePred in range(self.TotalClusters):
                """ if(np.isnan(self.centroids_pred[clasePred][0])):
                    centroid_similitudes.append((claseReal, clasePred,-1))
                else: """
                print(self.centroids_pred)
                similitude=cosine_similarity([self.centroids_true[claseReal]], [self.centroids_pred[clasePred]])
                centroid_similitudes.append((claseReal, clasePred,similitude))
        return centroid_similitudes
    def simpleLinkage(self):
        centroid_similitudes=[]
        for i in range(self.TotalClusters):
            vectorCondicion=[False for x in range(self.TotalDocumentos)]
            index=0
            for lbl in self.labels_pred:
                if i in lbl:
                    vectorCondicion[index]=True
                index+=1
            documentosXCluster=self.X_lsa[vectorCondicion]
            for j in range(self.TotalClases):
                vectorCondicion=[False for x in range(self.TotalDocumentos)]
                index=0
                for lbl in self.labels_true:
                    if j in lbl:
                        vectorCondicion[index]=True
                    index+=1
                documentosXClase=self.X_lsa[vectorCondicion]
                dist=[]
                for dxc in documentosXCluster:
                    for dxcl in documentosXClase:
                        d=cosine_similarity([dxc], [dxcl])
                        dist.append(d)
                dist.sort(reverse=True)
                centroid_similitudes.append((j,i,dist[0]))
        return centroid_similitudes   
    def completeLinkage(self):
        centroid_similitudes=[]
        for i in range(self.TotalClusters):
            vectorCondicion=[False for x in range(self.TotalDocumentos)]
            index=0
            for lbl in self.labels_pred:
                if i in lbl:
                    vectorCondicion[index]=True
                index+=1
            documentosXCluster=self.X_lsa[vectorCondicion]
            for j in range(self.TotalClases):
                vectorCondicion=[False for x in range(self.TotalDocumentos)]
                index=0
                for lbl in self.labels_true:
                    if j in lbl:
                        vectorCondicion[index]=True
                    index+=1
                documentosXClase=self.X_lsa[vectorCondicion]
                dist=[]
                for dxc in documentosXCluster:
                    for dxcl in documentosXClase:
                        d=cosine_similarity([dxc], [dxcl])
                        dist.append(d)
                dist.sort()
                centroid_similitudes.append((j,i,dist[0]))
        return centroid_similitudes  
    def averageLinkage(self):
        centroid_similitudes=[]
        for i in range(self.TotalClusters):
            vectorCondicion=[False for x in range(self.TotalDocumentos)]
            index=0
            for lbl in self.labels_pred:
                if i in lbl:
                    vectorCondicion[index]=True
                index+=1
            documentosXCluster=self.X_lsa[vectorCondicion]
            for j in range(self.TotalClases):
                vectorCondicion=[False for x in range(self.TotalDocumentos)]
                index=0
                for lbl in self.labels_true:
                    if j in lbl:
                        vectorCondicion[index]=True
                    index+=1
                documentosXClase=self.X_lsa[vectorCondicion]
                dist=[]
                for dxc in documentosXCluster:
                    for dxcl in documentosXClase:
                        d=cosine_similarity([dxc], [dxcl])
                        d=np.rad2deg(d)
                        dist.append(d)
                dist=np.array(dist)
                dist=dist.sum()/len(dist)
                centroid_similitudes.append((j,i,dist))
        return centroid_similitudes
    def MapClasses(self, nivel, current, error):
        # nodo hoja
        if nivel == self.TotalClusters:
            if error < self.Best[self.TotalClusters+1]:
                self.Best[self.TotalClusters+1] = error
                for i in range(self.TotalClusters):
                    self.Best[i] = current[i]
        else:
            # Si es un cluster vacio se debe ignorara
            if self.TotalesPorCluster[nivel] == 0:
                current[nivel] = -1 # cluster ignorado
                self.MapClasses(nivel + 1, current, error)
            else:
                # Primero trata este cluster sin ninguna asignación de clase
                current[nivel] = -1 # cluster asignado a ninguna clase (esto significa que tiene error total)
                self.MapClasses(nivel + 1, current, error + self.TotalesPorCluster[nivel])
                # Bucle para recorrer todas las clases en el cluster
                for i in range(len(self.MatrixDeConfusion[0])):
                    if self.MatrixDeConfusion[nivel][i] > 0:
                        ok = True
                        # chequeo para saber si esta clase ya ha sido asignada 
                        for j in range(nivel):
                            if current[j] == i:
                                ok = False
                                break
                        if ok:
                            current[nivel] = i
                            self.MapClasses(nivel + 1, current, (error + (self.TotalesPorCluster[nivel] - self.MatrixDeConfusion[nivel][i])))

    def CalcularMedidas(self):
        try:
            self.PrecisionPonderada = 0.0
            self.RecuerdoPonderado = 0.0
            self.Nicc = 0
            for i in range(self.TotalClusters+1):
                if self.Best[i] == -1: continue
                relevantes = self.MatrixDeConfusion[i][int(self.Best[i])]
                recuperados = self.TotalesPorCluster[i]
                totalrelevantes = self.TotalesPorClase[int(self.Best[i])]
                self.Nicc += relevantes
                estaPrecision=0
                if(recuperados!=0):
                    estaPrecision = relevantes * 100.0 / recuperados
                esteRecuerdo=0
                if(totalrelevantes!=0):
                    esteRecuerdo = relevantes * 100.0 / totalrelevantes
                estaMedidaF=0
                if((estaPrecision + esteRecuerdo)!=0):
                    estaMedidaF = (2.0 * estaPrecision * esteRecuerdo) / (estaPrecision + esteRecuerdo)
                self.PrecisionPonderada += estaPrecision * totalrelevantes
                self.RecuerdoPonderado += esteRecuerdo * totalrelevantes
                self.MedidaFPonderada += estaMedidaF * totalrelevantes

            self.PrecisionPonderada = self.PrecisionPonderada / np.sum(self.TotalesPorCluster)
            self.RecuerdoPonderado = self.RecuerdoPonderado / np.sum(self.TotalesPorCluster)
            self.MedidaFPonderada = self.MedidaFPonderada / np.sum(self.TotalesPorCluster)
            self.Picc = self.Nicc * 100.0 / np.sum(self.TotalesPorCluster)
            self.Wtpr = self.WeightedTruePositiveRate()
            self.Wfpr = self.WeightedFalsePositiveRate()
            self.WSpecificity = 1 - self.Wfpr
            self.WAccuracy = self.WeightedAccuracy()
            self.Wnpv = self.WeightedNPV()
            if(self.Wnpv==None):
                self.Wnpv=0
            self.Wfdr = self.WeightedFDR()
            self.TP=0
            self.TN=0
            self.FP=0
            self.FN=0
            for i in range(self.TotalClases+1):
                self.TP+=self.TruePositive(i)
                aux=self.TrueNegative(i)
                self.TN+=aux
                self.FP+=self.FalsePositive(i)
                self.FN+=self.FalseNegative(i)
        except Exception as e1:
            print(e1)
            
    def TruePositive(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters+1):
            if self.Best[i] == classIndex:
                correct += self.MatrixDeConfusion[i][classIndex]
        return correct

    def TrueNegative(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters+1):
            if self.Best[i] != classIndex:
                for j in range(self.TotalClases+1):
                    if j != classIndex:
                        correct += self.MatrixDeConfusion[i][j]
        return correct
    """ def TrueNegative(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters+1):
            if self.Best[i] != classIndex:
                for j in range(self.TotalClases+1):
                    if j == self.Best[i]:
                        correct += self.MatrixDeConfusion[i][j]
        return correct """
    def FalsePositive(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters+1):
            if self.Best[i] == classIndex:
                for j in range(self.TotalClases+1):
                    if j != classIndex:
                        correct += self.MatrixDeConfusion[i][j]
        return correct

    def FalseNegative(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters+1):
            if self.Best[i] != classIndex:
                correct += self.MatrixDeConfusion[i][classIndex]
        return correct
    def WeightedTruePositiveRate(self):
        truePosTotal = 0
        for j in range(self.TotalClases):
            temp = self.TruePositiveRate(j)
            truePosTotal += (temp * self.TotalesPorClase[j])
        return truePosTotal / np.sum(self.TotalesPorCluster)
    def TruePositiveRate(self, classIndex):
        tp = self.TruePositive(classIndex)
        if self.TotalesPorClase[classIndex] == 0: return 0
        return tp / self.TotalesPorClase[classIndex]
    def FalsePositiveRate(self, classIndex):
        fp = self.FalsePositive(classIndex)

        if np.sum(self.TotalesPorCluster) - self.TotalesPorClase[classIndex] == 0: return 0
        return fp / (np.sum(self.TotalesPorCluster) - self.TotalesPorClase[classIndex])

    def WeightedFalsePositiveRate(self):
        trueNegTotal = 0
        for j in range(self.TotalClases):
            temp = self.FalsePositiveRate(j)
            trueNegTotal += (temp * self.TotalesPorClase[j])
        return trueNegTotal / np.sum(self.TotalesPorCluster)
        

    def AccuracyRate(self, classIndex):
        tp = self.TruePositive(classIndex)
        fp = self.TrueNegative(classIndex)
        #return (tp + fp) / self.TotalDocumentos
        return (tp +fp )/np.sum(self.TotalesPorCluster)

    def WeightedAccuracy(self):
        trueAccTotal = 0
        for j in range(self.TotalClases):
            
            temp = self.AccuracyRate(j)
            trueAccTotal += (temp * self.TotalesPorClase[j])           
        return trueAccTotal / np.sum(self.TotalesPorCluster)

    def NPVRate(self, classIndex):
        tn = self.TrueNegative(classIndex)
        fn = self.FalseNegative(classIndex)
        if tn + fn == 0: return 0
        return tn / (tn + fn)

    def WeightedNPV(self):
        trueNpvTotal = 0
        for j in range(self.TotalClases):
            temp = self.NPVRate(j)
            trueNpvTotal += (temp * self.TotalesPorClase[j])
    def FDRRate(self, classIndex):
        fp = self.FalsePositive(classIndex)
        tp = self.TrueNegative(classIndex)
        if fp + tp == 0: return 0
        return fp / (fp + tp)

    def WeightedFDR(self):
        trueFdrTotal = 0
        for j in range(self.TotalClases):
            temp = self.FDRRate(j)
            trueFdrTotal += (temp * self.TotalesPorClase[j])
        return trueFdrTotal / np.sum(self.TotalesPorCluster)
    def getMetrics(self):
        return self.PrecisionPonderada, self.RecuerdoPonderado, self.MedidaFPonderada, self.Nicc, self.Picc,self.Wtpr,self.Wfpr, self.WAccuracy, self.Wnpv,self.Wfdr,self.TP,self.TN,self.FP,self.FN
    def __str__(self):
        reporte = ""
        for i in range(self.TotalClusters):
            if self.Best[i] == -1: continue
            relevantes = self.MatrixDeConfusion[i][int(self.Best[i])]
            reporte += "Cluster " + str(i) + " to class " + str(int(self.Best[i])) + " con " + str(relevantes) + " docs. \n"
        
        reporte += "Matrix de Confusion \n"
        for i in range(self.TotalClusters):
            for j in range(self.TotalClases):
                reporte += str(self.MatrixDeConfusion[i][j]) + "\t"
            reporte += "\n"
        reporte += "Weighted Avg. Precision = " + str(self.PrecisionPonderada) + "\n"
        reporte += "Weighted Avg. Recuerdo  = " + str(self.RecuerdoPonderado) + "\n"
        reporte += "Weighted Avg. F-measure = " + str(self.MedidaFPonderada) + "\n"
        reporte += "Instancias correctamente agrupadas = " + str(self.Nicc) + "\n"
        reporte += "% Instancias correctamente agrupadas = " + str(self.Picc) + "\n"
        reporte += "Rata ponderada de verdaderos positivos =" + str(self.Wtpr) + "\n"
        reporte += "Rata ponderada de falsos positivos =" + str(self.Wfpr) + "\n"
        reporte += "Exactitud promedio ponderada (Accuracy) =" + str(self.WAccuracy) + "\n"
        reporte += "Negative predictive value (NPV) =" + str(self.Wnpv) + "\n"
        reporte += "Rata ponderada de falsos descubrimientos (FDR) =" + str(self.Wfdr) + "\n"
        reporte += "Verdaderos Positivos= "+str(self.TP)+ "\n"
        reporte += "Verdaderos Negativos= "+str(self.TN)+ "\n"
        reporte += "Falsos Positivos= "+str(self.FP)+ "\n"
        reporte += "Falsos Negativos= "+str(self.FN)+ "\n"
        reporte += "Precision tp/(tp+fp)= "+str(self.TP/(self.TP+self.FP))+" \n"
        return reporte
        
        
