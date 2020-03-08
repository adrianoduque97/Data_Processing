import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Normalizar
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


caracteristicas = [4,5,28, 48, 64, 105, 128, 161,151,153,192,241, 281, 318, 336, 338, 378, 433,442, 451, 453,455, 472, 475, 493]
vecinos =[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]


def generarMatriz(matriz,comb):
    a = []
    for cols in range(len(matriz.columns)):
        if cols in comb:
            a.append(matriz[cols])
    a= pd.DataFrame(a)
    a= np.transpose(a)
    print(a)
    return a

def pcaUse(matrizTest, matrizValid):
    pca = PCA(n_components=20, svd_solver='auto')
    pca.fit(matrizTest)
    matrizTest = pca.transform(matrizTest)
    matrizValid = pca.transform(matrizValid)

    return matrizTest,matrizValid

if __name__ == '__main__':


    labels= pd.read_csv("madelon_train.labels",header=None)
    labels = labels.to_numpy()
    doc= pd.read_csv("madelon_train.data", delimiter=" " , header=None)

    validLabels=pd.read_csv("madelon_valid.labels",header=None)
    valdidDoc=pd.read_csv("madelon_valid.data", delimiter=" " , header=None)
    validLabels=validLabels.to_numpy()


    normalizador = Normalizar.Normalizar()
    normalizador.normalizar("madelon_train.data")
    normalizador.normalizar("madelon_valid.data")

    doc.pop(500)
    valdidDoc.pop(500)
    print("Matriz test:\n"+str(doc))
    print("Matriz Valid: \n "+str(valdidDoc))

    norm = pd.read_csv("normalizado_madelon_train.data", header=None,delimiter=" ")
    normvalidate = pd.read_csv("normalizado_madelon_valid.data",header=None, delimiter=" ")
    print("Matriz Normalizada Train:\n"+str(norm))
    print("Matriz Normalizada Valid:\n" + str(normvalidate))

    matrizTest= generarMatriz(norm,caracteristicas)
    matrizValid= generarMatriz(normvalidate,caracteristicas)

    matrizTest,matrizValid = pcaUse(matrizTest,matrizValid)



    manhattan=[]
    euclidean=[]

    print("MANHATTAN DISTANCE")
    for i in vecinos:
        n1 = KNeighborsClassifier(n_neighbors=i,weights="distance",metric="manhattan", algorithm='auto', leaf_size=70)
        n1.fit(matrizTest,labels.ravel())
        scoreTest =n1.score(matrizTest, labels.ravel())
        scoreValid=n1.score(matrizValid, validLabels.ravel())
        manhattan.append(scoreValid)
        print("Accuracy evaluando sobre test, con "+str(i)+" vecinos:" + "\n"+str(scoreTest))
        print("Accuracy evaluando sobre Madeolon_Valid, con " +str(i)+" vecinos:" +"\n"+str(scoreValid))

    print("EUCLIDEAN DISTANCE")
    for i in vecinos:
        n1 = KNeighborsClassifier(n_neighbors=i,weights="distance" ,metric='euclidean' , algorithm='auto',leaf_size=70)
        n1.fit(matrizTest, labels.ravel())
        scoreTestE = n1.score(matrizTest, labels.ravel())
        scoreValidE = n1.score(matrizValid, validLabels.ravel())
        euclidean.append(scoreValidE)
        print("Accuracy evaluando sobre test, con " + str(i) + " vecinos:" + "\n" + str(scoreTestE))
        print("Accuracy evaluando sobre Madeolon_Valid, con " + str(i) + " vecinos:" + "\n" + str(scoreValidE))

    plt.plot(vecinos,manhattan, label='Manhattan')
    plt.plot(vecinos,euclidean, label= 'Euclidean')
    plt.legend()
    plt.xlabel("# Vecinos")
    plt.ylabel("Accuracy")
    plt.title("Graph testing values on KNN")
    fig = plt.gcf()
    fig.canvas.set_window_title('Project 5')

    plt.show()

