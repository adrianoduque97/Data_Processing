import  numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
import Normalizar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from itertools import chain, combinations
import itertools

def normalize(matriz):
    scaler = MinMaxScaler()
    print(scaler.fit(matriz))

    normalizado = scaler.transform(matriz)
    norm = pd.DataFrame(data=normalizado)
    norm.to_csv('Normal2.data', header=None, index=False)
    return norm


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def findsubsets(s, n):
    return list(itertools.combinations(s, n))
def findsubsetsRamgo(s, n,m):
    for x in range(n,m):
        k=list(itertools.combinations(s, n))
    return k

def correlationCHI(matriz, labels):
    selector = SelectKBest(chi2, k=50).fit(matriz, labels)
    # sel2 = SelectKBest(chi2,k=50).fit_transform()
    print(selector)

    X_new = selector.transform(matriz)  # SelectKBest(chi2,k=4 ).fit_transform(norm, labels)
    X_new = pd.DataFrame(X_new)
    X_new.to_csv('NormalCHI.data', header=None, index=False)
    return X_new


def correlationPearson(df):
    cor = df.corr()
    print("Correlacion Pearson:\n"+str(cor))

    cor_target = abs(cor)  # Selecting highly correlated features

    non_rel= cor_target[cor_target<0.0000000]
    non_rel=pd.DataFrame(non_rel)
    print(non_rel)

    relevant_features = cor_target[cor_target > 0.6]
    relevant_features = pd.DataFrame(relevant_features)
    relevant_features.to_csv('normalPearson.data', header=None, index=False)
    print("Relevant:\n"+str(relevant_features))
    arr={}
    a=[]
    for col in range(len(relevant_features.columns)):
        arr[col]=0

    for cols in range(len(relevant_features.columns)):
        for fila in range(len(relevant_features)):
            if not math.isnan(relevant_features[cols][fila]) and cols!=fila:
                arr[cols]+=fila

    for l in range(len(non_rel.columns)):
        for f in range(len(non_rel)):
            if not math.isnan(non_rel[l][f]) and l != f:
                arr[l] += f


    a=[]
    for key in arr.keys():
        if arr[key]>1:
            a.append(key)

    print("Caracteristicas resultantes para Hipotesis:"+str(len(a))+"\n"+str(a))
    return a


def potencia(c):
    if len(c) == 0:
        return [[]]
    r = potencia(c[:-1])
    return r + [s + [c[-1]] for s in r]


def quitarPotencia(matriz):
    temporal=[]
    for element in matriz:
        if len(element)>17 :
            temporal.append(element)

    return temporal


def SVM(matriz):
    svm = SVC(kernel='rbf', gamma='auto',C=0.025, random_state=101)
    svm.fit(matriz, labels.ravel())
    score=svm.score(matriz, labels.ravel())
    print(svm.score(matriz, labels.ravel()))
    return score


def knn(matriz):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(matriz, labels.ravel())
    score=neigh.score(matriz, labels.ravel())
    print("Accuracy de caracteristicas seleccionadas: "+str(neigh.score(matriz, labels.ravel())))
    return score


def bayes(matriz):
    bayes = MultinomialNB()
    bayes.fit(matriz, labels.ravel())
    score=bayes.score(matriz,labels.ravel())
    print(bayes.score(matriz, labels.ravel()))
    return score


def generarMatriz(matriz,comb):
    a = []
    for cols in range(len(matriz.columns)):
        if cols in comb:
            a.append(matriz[cols])
    a= pd.DataFrame(a)
    a= np.transpose(a)
    print(a)
    return a


def wrapper(pos,norm):
    pre = 0
    f = []
    for i in pos:
        matG = generarMatriz(norm, i)
        kn = knn(matG)
        if (kn > pre):
            pre = kn
            f = i

    print(" Cracteristicas Seleccionadas:" + str(f)+"\nAccuracy: " + str(pre) )

    return f


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

    norm = pd.read_csv("normalizado_madelon_train.data", header=None,delimiter=" ")
    #norm = normalize(doc)
    normvalidate = pd.read_csv("normalizado_madelon_valid.data",header=None, delimiter=" ")
    print("Matriz Normalizada\n"+str(norm))

    nomrCHI = correlationCHI(norm, labels)
    print("Matriz resultante correlacion CHI cuadrado\n" + str(nomrCHI))
    print("Caracteristicas mas Relevantes "+ str(len(nomrCHI.columns)))

    normPearson = correlationPearson(norm)

    posibles = findsubsets(normPearson,20)
    posibles= list(posibles)
    printT=pd.DataFrame(posibles)
    printT.to_csv('posibles.csv', header=None, index=False)
    print("Posibles espacios de busqueda: "+str(len(posibles)))
    pos=quitarPotencia(posibles)

    caracteristicas=wrapper(pos,norm)

    matriz= generarMatriz(norm,caracteristicas)
    mat2= generarMatriz(normvalidate,caracteristicas)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(matriz,labels.ravel())

    print("CARACTERISTICAS SELECCIONADAS:"+str(len(caracteristicas))+"\n"+str(caracteristicas))
    print("Accuracy evaluando sobre test: "+str(neigh.score(matriz,labels.ravel())))
    print("Accuracy evaluando sobre Madeolon_Valid: "+str(neigh.score(mat2,validLabels.ravel())))
    print(neigh.predict(mat2))


    p1 =generarMatriz(norm,[28,48,64,105,128,151,153,241,281,318,336,338,378,433,451,453,472,475,493])
    p2 = generarMatriz(normvalidate, [28, 48, 64, 105, 128, 151, 153, 241, 281, 318, 336, 338, 378, 433, 451, 453, 472, 475, 493])
    n1 = KNeighborsClassifier(n_neighbors=3)
    n1.fit(p1, labels.ravel())
    print("Accuracy evaluando sobre test: " + str(n1.score(p1, labels.ravel())))
    print("Accuracy evaluando sobre Madeolon_Valid: " + str(n1.score(p2, validLabels.ravel())))
    







