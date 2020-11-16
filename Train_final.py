import datetime
ts1=datetime.datetime.now().timestamp()
########## Modelos ###########
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

######### Modelo_Multiclases #######
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import confusion_matrix, plot_confusion_matrix,f1_score,recall_score
from sklearn.inspection import permutation_importance

######### STANDAR ############
import numpy as np
import pandas as pd 
import Estadisticos
import matplotlib.pylab as plt


###########################
#           DATA          #
###########################
Data=pd.read_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/datos_Final_entrenamiento.csv", sep=',')
y=Data.TIPO
X=Data.drop('TIPO',axis=1).drop("Estrella",axis=1).replace(np.nan,0).drop("No",axis=1)
# nombres=["ABBE","MAD","OCTILa","OCTILb","SKEW","MEDIAN"]#,
nombres_no_usados=["MADMAD"]#"PITAR","RULD","TFACT2","TFACT","DIFER","INTEGRAL","INTEGRAL2","MADMAD","DERPROM","DIFDER","PROY","PROY2","MEGAMAD","TANGENTE","DERMED","CONTPOS"]
nombres_no_usados_Random=["MADMAD","MAD","TFACT2","DIFER","INTEGRAL","INTEGRAL2","DERPROM","DIFDER","PROY2","MEGAMAD","TANGENTE","DERMED","CONTPOS","ABBE","OCTILa","OCTILb","SKEW","MEDIAN"]
nombres_no_usados_Knbrs=["MADMAD","RULD","TFACT2","TFACT","DIFER","DIFDER","PROY","PROY2","MEGAMAD","TANGENTE","DERMED","CONTPOS","ABBE","MAD","OCTILa","SKEW"]
nombres_no_usados_Vector=["MADMAD","PITAR","RULD","TFACT2","TFACT","DIFER","INTEGRAL","INTEGRAL2","DERPROM","DIFDER","PROY","MEGAMAD","DERMED","CONTPOS","ABBE","MEDIAN"]
nombres_no_usados_Tree=["MADMAD","RULD","TFACT2","DIFER","INTEGRAL","INTEGRAL2","DERPROM","DIFDER","PROY","MEGAMAD","TANGENTE","DERMED","CONTPOS","ABBE","OCTILa","SKEW"]
for nombre in nombres_no_usados:
    X=X.drop(nombre,axis=1)
Features_usados=list(X.columns) 

print("-------------------------------------------")
print("-     Data training - Time milestones     -")
print("-------------------------------------------")
print("Used features:")
print(Features_usados)
Test_size=0.3
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=Test_size)
ts_data=datetime.datetime.now().timestamp()
print("Data done:"+str(ts_data-ts1)+" Seconds")


###########################
#         MODELOS         #
###########################

modelos={
    'Random':RandomForestClassifier(max_depth=11,max_features=3,n_estimators=3),
    'Tree':tree.DecisionTreeClassifier(max_depth=9,max_features=4,criterion="entropy"),#Max_features10->ideal
    'Vector':svm.SVC(kernel="linear"),
    'Knbrs':KNeighborsClassifier(n_neighbors=11,p=1)
    }
Nombres_modelos={"Random":"Random forest","Tree":"Decision tree","Vector":"Support vector machine"}#,"Knbrs":"K-neighbors"}
for modelo in modelos.keys():
    modelos[modelo].fit(X_train,y_train)
    ts_FIT=datetime.datetime.now().timestamp()
    print(modelo+"Fit done:"+str(ts_FIT-ts1)+" Seconds")

############################
#     Confusion matrix     #
############################
def confusion(modelo,nombre):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.title(nombre + " confusion matrix")
    plot_confusion_matrix(modelo, X_test, y_test, normalize ='true',ax=ax,cmap="Reds")
    plt.savefig("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/ALL_STARS_ALL/conf_matrix_All_"+nombre+"_2.jpg") 
#Generación de matrices de confusion con metodo (Descomentar y cambiar titulo y path)
for modelo in modelos.keys():
    confusion(modelos[modelo],Nombres_modelos[modelo])

ts_figs=datetime.datetime.now().timestamp()
print("Figs done:"+str(ts_figs-ts1)+" Seconds")

###########################
#   IMPORTANCES SORTED   #
###########################

importance=modelos["Tree"].feature_importances_
Tree_Sorted={}
for i,v in enumerate(importance):
    Tree_Sorted[Features_usados[i]]=np.abs(v)

importance=modelos["Random"].feature_importances_
Random_Sorted={}
for i,v in enumerate(importance):
    Random_Sorted[Features_usados[i]]=np.abs(v)

results = permutation_importance(modelos["Knbrs"], X_train, y_train, scoring='accuracy')
importance = results.importances_mean
Knbrs_Sorted={}
for i,v in enumerate(importance):
    Knbrs_Sorted[Features_usados[i]]=np.abs(v)

Vector_Sorted={}
importance = modelos["Vector"].coef_[0]
for i,v in enumerate(importance):
    Vector_Sorted[Features_usados[i]]=np.abs(v)

Tree_Sorted={k: v for k, v in sorted(Tree_Sorted.items(), key=lambda item: item[1])}
Random_Sorted={k: v for k, v in sorted(Random_Sorted.items(), key=lambda item: item[1])}
Knbrs_Sorted={k: v for k, v in sorted(Knbrs_Sorted.items(), key=lambda item: item[1])}
Vector_Sorted={k: v for k, v in sorted(Vector_Sorted.items(), key=lambda item: item[1])}
Sorteados={
"Tree":Tree_Sorted,
"Random":Random_Sorted,
"Knbrs":Knbrs_Sorted,
"Vector":Vector_Sorted
}

print("-------------------------------------------")
print("-       Feature importance by model       -")
print("-------------------------------------------")

def sorter(dic):
    dic=[(k, v) for k, v in sorted(dic.items(), key=lambda item: item[1])]
    return dic

for sorteado in Sorteados.keys():
    acum=0
    print(Nombres_modelos[sorteado]+" sorted importance:")
    print("Feature \t| Importance \t| Accumulated ")
    for i,v in sorter(Sorteados[sorteado]):
        acum+=v
        print("%s \t| %.5f \t|  %.5f" %(i,v,acum))
    print("--------------------------")



ts_importance=datetime.datetime.now().timestamp()
print("Importances done: "+str(ts_importance-ts1)+" Seconds")

###########################
#      SCORE ON TEST      #
###########################
print("-------------------------------------------")
print("-    Score obtained in the test dataset   -")
print("-------------------------------------------")


for modelo in modelos.keys():
    print("-----------------------")
    print("Score %s :"%Nombres_modelos[modelo])
    print("Accuracy : %.5f" %modelos[modelo].score(X_test,y_test))
    print("F1 score : %.5f" %f1_score(modelos[modelo].predict(X_test),y_test,average="weighted") )
    print("Recall : %.5f" %recall_score(modelos[modelo].predict(X_test),y_test,average="weighted"))
print("-----------------------")

ts_Score=datetime.datetime.now().timestamp()
print("Score done: "+str(ts_Score-ts1)+" Seconds")