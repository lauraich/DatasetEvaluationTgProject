import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
import pandas as pd
class Principal:
    def __init__(self, labels_true,labels_pred, cantidadDeClases, cantidadDeClusters, data):
        self.data=data
        self.labels_true=labels_true
        self.labels_pred=labels_pred
        lsa = make_pipeline(TruncatedSVD(n_components=100))
        self.X_lsa = lsa.fit_transform(self.data)
        matrizdatosdetallados = np.zeros(shape=(len(labels_true), 3), dtype=int)
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

        # Convert the matrix to a numpy array for easier manipulation
        self.MatrixDeConfusion = np.zeros((self.TotalClusters, self.TotalClases), dtype=int)
        for i in range(self.TotalDocumentos):
            idDocumento, idClaseReal, idGrupoEncontrado = matrizdatosdetallados[i]
            self.MatrixDeConfusion[idGrupoEncontrado][idClaseReal] += 1

        # Calculate the totals by cluster and class
        self.TotalesPorCluster = np.sum(self.MatrixDeConfusion, axis=1)
        self.TotalesPorClase = np.sum(self.MatrixDeConfusion, axis=0)

        # Initialize the best and current arrays
        self.Best = np.full(shape=self.TotalClusters + 1, dtype=float,fill_value=-1)
        self.current = np.zeros(self.TotalClusters + 1)
        self.Best[self.TotalClusters] = float('inf')

        # Map the classes and calculate the measures
        self.MapClasses2()
        self.CalcularMedidas()
    def calcularCentroides(self):
        self.centroids_true = []
        for i in range(self.TotalClases):
            centroid = self.X_lsa[self.labels_true == i].mean(axis=0)
            self.centroids_true.append(centroid)
        self.centroids_pred = []
        for i in range(self.TotalClusters):
            centroid = self.X_lsa[self.labels_pred == i].mean(axis=0)
            self.centroids_pred.append(centroid)
    def MapClasses2(self):
        self.calcularCentroides()
        centroid_similitudes=[]
        for claseReal in range(self.TotalClases):
            for clasePred in range(self.TotalClusters):
                """ if(np.isnan(self.centroids_pred[clasePred][0])):
                    centroid_similitudes.append((claseReal, clasePred,-1))
                else: """
                similitude=cosine_similarity([self.centroids_true[claseReal]], [self.centroids_pred[clasePred]])
                centroid_similitudes.append((claseReal, clasePred,similitude))
        band=True
        while(len(centroid_similitudes)>0):
            centroid_similitudes=sorted(centroid_similitudes, key=lambda x: x[2], reverse=True)
            if(band):
                pand=pd.DataFrame(centroid_similitudes)
                pand.to_csv("centroid_similitudes.csv")
                band=False
            claseReal, clasePred, similitude=centroid_similitudes[0]
            self.Best[clasePred]=claseReal
            centroid_similitudes=[x for x in centroid_similitudes if x[0]!=claseReal and x[1]!=clasePred]
            
    def MapClasses(self, nivel, current, error):
        # nodo hoja
        if nivel == self.TotalClusters:
            if error < self.Best[self.TotalClusters]:
                self.Best[self.TotalClusters] = error
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
            for i in range(self.TotalClusters):
                if self.Best[i] == -1: continue
                relevantes = self.MatrixDeConfusion[i][int(self.Best[i])]
                recuperados = self.TotalesPorCluster[i]
                totalrelevantes = self.TotalesPorClase[int(self.Best[i])]
                self.Nicc += relevantes

                estaPrecision = relevantes * 100.0 / recuperados
                esteRecuerdo = relevantes * 100.0 / totalrelevantes
                estaMedidaF = (2.0 * estaPrecision * esteRecuerdo) / (estaPrecision + esteRecuerdo)

                self.PrecisionPonderada += estaPrecision * totalrelevantes
                self.RecuerdoPonderado += esteRecuerdo * totalrelevantes
                self.MedidaFPonderada += estaMedidaF * totalrelevantes

            self.PrecisionPonderada = self.PrecisionPonderada / self.TotalDocumentos
            self.RecuerdoPonderado = self.RecuerdoPonderado / self.TotalDocumentos
            self.MedidaFPonderada = self.MedidaFPonderada / self.TotalDocumentos
            self.Picc = self.Nicc * 100.0 / self.TotalDocumentos
            self.Wtpr = self.WeightedTruePositiveRate()
            self.Wfpr = self.WeightedFalsePositiveRate()
            self.WSpecificity = 1 - self.Wfpr
            self.WAccuracy = self.WeightedAccuracy()
            self.Wnpv = self.WeightedNPV()
            self.Wfdr = self.WeightedFDR()
        except Exception as e1:
            print(e1.message)
            
    def TruePositive(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters):
            if self.Best[i] == classIndex:
                correct += self.MatrixDeConfusion[i][classIndex]
        return correct

    def TrueNegative(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters):
            if self.Best[i] != classIndex:
                for j in range(self.TotalClases):
                    if j != classIndex:
                        correct += self.MatrixDeConfusion[i][j]
        return correct

    def FalsePositive(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters):
            if self.Best[i] == classIndex:
                for j in range(self.TotalClases):
                    if j != classIndex:
                        correct += self.MatrixDeConfusion[i][j]
        return correct

    def FalseNegative(self, classIndex):
        correct = 0.0
        for i in range(self.TotalClusters):
            if self.Best[i] != classIndex:
                correct += self.MatrixDeConfusion[i][classIndex]
        return correct
    def WeightedTruePositiveRate(self):
        truePosTotal = 0
        for j in range(self.TotalClases):
            temp = self.TruePositiveRate(j)
            truePosTotal += (temp * self.TotalesPorClase[j])
        return truePosTotal / self.TotalDocumentos
    def TruePositiveRate(self, classIndex):
        tp = self.TruePositive(classIndex)
        if self.TotalesPorClase[classIndex] == 0: return 0
        return tp / self.TotalesPorClase[classIndex]
    def FalsePositiveRate(self, classIndex):
        fp = self.FalsePositive(classIndex)

        if self.TotalDocumentos - self.TotalesPorClase[classIndex] == 0: return 0
        return fp / (self.TotalDocumentos - self.TotalesPorClase[classIndex])

    def WeightedFalsePositiveRate(self):
        trueNegTotal = 0
        for j in range(self.TotalClases):
            temp = self.FalsePositiveRate(j)
            trueNegTotal += (temp * self.TotalesPorClase[j])
        return trueNegTotal / self.TotalDocumentos

    def AccuracyRate(self, classIndex):
        tp = self.TruePositive(classIndex)
        fp = self.TrueNegative(classIndex)
        return (tp + fp) / self.TotalDocumentos

    def WeightedAccuracy(self):
        trueAccTotal = 0
        for j in range(self.TotalClases):
            temp = self.AccuracyRate(j)
            trueAccTotal += (temp * self.TotalesPorClase[j])
        return trueAccTotal / self.TotalDocumentos

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
        return trueFdrTotal / self.TotalDocumentos

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
        return reporte
        