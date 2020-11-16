########## Modelos ###########
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

######### Best_Params_Search #######
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

######### STANDAR ############
import numpy as np
import pandas as pd 
import Estadisticos
import matplotlib.pylab as plt
import datetime

#Timestamp0
ts1=datetime.datetime.now().timestamp()


###########################
#           DATA          #
###########################
Data=pd.read_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/datos_intento.csv", sep=',')
y=Data.TIPO
X=Data.drop('TIPO',axis=1).drop("Estrella",axis=1).replace(np.nan,0).drop("No",axis=1)
Test_size=0.8
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state =0,test_size=Test_size)

#Timestamp1
ts_data=datetime.datetime.now().timestamp()
print("Data listo:"+str(ts_data-ts1)+" Segundos")

###########################
#         MODELOS         #
###########################
VectorModel=svm.SVC()    #  .get_params().keys())  
TreeModel=tree.DecisionTreeClassifier() #.get_params().keys())
RandomModel=RandomForestClassifier()    #get_params().keys())
KnbrsModel=KNeighborsClassifier()       #.get_params().keys())


##----# Tuned parameters with GridSearchCV #-----##
Random_tuned_parameters = [{'max_depth' : [11,12,13], 'max_features' : [3,4,5,6], 'n_estimators' : [1,2,3]}]
Tree_tuned_parameters = [{'max_depth' : [8,9,10,11], 'max_features' : [6,7,8,9,10],'criterion':['entropy']}]
Vector_tuned_parameters = [{'kernel':["linear"]}]
Knbrs_tuned_parameters = [{'n_neighbors' : [11,12,13], 'p' :[1]}]

n_jobs=-1
scoring='balanced_accuracy'
##----# Grid of parameters #----##
RandomTreeGrid = GridSearchCV(RandomModel,Random_tuned_parameters, scoring=scoring, n_jobs=n_jobs, cv=5, verbose=1,refit=True)
TreeModelGrid= GridSearchCV(TreeModel,Tree_tuned_parameters, scoring=scoring,n_jobs=n_jobs,cv=5,verbose=1,refit=True)
VectorGrid=GridSearchCV(VectorModel,Vector_tuned_parameters,scoring=scoring,n_jobs=n_jobs,cv=5,verbose=1,refit=True)
KNbrsGrid= GridSearchCV(KnbrsModel,Knbrs_tuned_parameters, scoring=scoring,n_jobs=n_jobs,cv=5,verbose=1,refit=True)
modelos={'Random':RandomTreeGrid,'Tree':TreeModelGrid,'Vector':VectorGrid,'Knbrs':KNbrsGrid}

##----# Fitting all models and printing params #----##
for modelo in modelos.keys():
    modelos[modelo].fit(X_train,y_train)
    print("--------------------------")
    print("|Best "+modelo+" Params:")
    print(modelos[modelo].best_params_)
print("--------------------------")

#Timestamp2
ts_FIT=datetime.datetime.now().timestamp()
print("Fit listo:"+str(ts_FIT-ts1)+" Segundos")

############################
#     Confusion matrix     #
############################
def confusion(modelo):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.title(modelo + " confusion matrix")
    plot_confusion_matrix(modelos[modelo], X_test, y_test, normalize ='true',ax=ax,cmap="Reds")
    plt.savefig("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/conf_matrix_"+modelo+".jpg") 

for modelo in modelos.keys():
    confusion(modelo)

#Timestamp3
ts_figs=datetime.datetime.now().timestamp()
print("Figs listo:"+str(ts_figs-ts1)+" Segundos")

