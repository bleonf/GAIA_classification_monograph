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
from sklearn.metrics import confusion_matrix, plot_confusion_matrix,f1_score,recall_score
from sklearn.inspection import permutation_importance

######### STANDAR ############
import numpy as np
import pandas as pd 
import Estadisticos
import matplotlib.pylab as plt


#-------------------#
#     All Inputs    #
#-------------------#
print("Choose classifier:")
print(" a=Random Forest Classifier \n b=Decision Tree \n c=Support Vector Machine \n d=K-Neighbors")
classifier_input=input("Enter a,b,c or d: ")

print("Number of top features to use (1-21)")
number_of_features=int(input("If 0, custom features will be used:"))

save_input=input("Do you want to save figure? (Y/N): ")
show_input=input("Do you want to show figure? (Y/N): ")

#-----------------------------#
#       DATA GENERATION       #
#-----------------------------#
All_Features=["MEDIAN","PITAR","MAD","TFACT","SKEW","PROY2","MEGAMAD","PROY","OCTILb","RULD","ABBE","INTEGRAL","TFACT2","DERMED","INTEGRAL2","TANGENTE","OCTILa","CONTPOS","DERPROM","DIFER","DIFDER"]

Features=All_Features[:number_of_features]
if number_of_features==0: #Custom_features
    Features=["MEDIAN","ABBE","OCTILa","OCTILb","SKEW","MAD"]
Characters=["Estrella","TIPO"]
columnas=Features+Characters
# Star_names=np.genfromtxt("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/GAIA_names.txt")
TOTAL=pd.read_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/datos_Final_entrenamiento.csv", usecols=columnas)
pdT2=TOTAL[TOTAL["TIPO"] == "T2"]         #287
pdCEP=TOTAL[TOTAL["TIPO"] == "CEP"]       #4626
pdACEP=TOTAL[TOTAL["TIPO"] == "ACEP"]     #144
pdECL=TOTAL[TOTAL["TIPO"] == "ECL"]       #1043
pdLPV=TOTAL[TOTAL["TIPO"] == "LPV"]       #5000
pdDSCUTI=TOTAL[TOTAL["TIPO"] == "DSCUTI"] #2788
pdBE=TOTAL[TOTAL["TIPO"] == "Be"]         #475
pdRR=TOTAL[TOTAL["TIPO"] == "RRLYRAE"]    #12527

N=200
random_pdT2=pdT2.sample(n=N)            
random_pdCEP=pdCEP.sample(n=N)        
random_pdACEP=pdACEP.sample(n=144)     
random_pdECL=pdECL.sample(n=N)     
random_pdLPV=pdLPV.sample(n=N)      
random_pdDSCUTI=pdDSCUTI.sample(n=N)
random_pdBE=pdBE.sample(n=N)         
random_pdRR=pdRR.sample(n=N)         

TRAIN=pd.concat([random_pdT2,random_pdCEP,random_pdACEP,random_pdECL,random_pdLPV,random_pdDSCUTI,random_pdBE,random_pdRR],ignore_index=True)
print(TRAIN.shape)

CLASSIFY=pd.read_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/datos_Final_GAIA.csv", usecols=columnas)
TIPOS=pd.read_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/GAIA_CLASSIFICATION.csv",usecols=Characters)
TIPOS["Estrella"]=TIPOS["Estrella"].astype(str)+".dat"

for estrella in CLASSIFY["Estrella"].tolist():
    CLASSIFY.loc[CLASSIFY["Estrella"]==estrella,"TIPO"]=str(TIPOS.loc[TIPOS["Estrella"]==estrella,"TIPO"].values[0])

X=TRAIN[Features].replace(np.nan,0)
y=TRAIN["TIPO"]
Test_size=0.15
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=Test_size)

ts_data=datetime.datetime.now().timestamp()
print("Data done:"+str(ts_data-ts1)+" Seconds\n\n")

#-------------------------------#
#              TRAIN            #
#-------------------------------#

if number_of_features <3 and number_of_features >0:
    Max_features=number_of_features
elif number_of_features==0:
    Max_features=len(Features)
else:
    Max_features=3

classiffiers={
"a":RandomForestClassifier(max_depth=11,max_features=Max_features,n_estimators=3),
"b":tree.DecisionTreeClassifier(max_depth=9,max_features=Max_features,criterion="entropy"),
"c":svm.SVC(kernel="linear"),
"d":KNeighborsClassifier(n_neighbors=11,p=1)}

names={"a":"Random Forest","b":"Decision Tree","c":"Support Vector Machines","d":"K-Neighbours"}

Random=classiffiers[classifier_input]

Random.fit(X_train,y_train)
ts_train=datetime.datetime.now().timestamp()
print("Train done:"+str(ts_train-ts1)+" Seconds\n\n")

#-------------------------------#
#       Confusion Matrix        #
#-------------------------------#
def confusion(modelo,nombre,show="N",save="N"):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.title(nombre + " confusion matrix")
    plot_confusion_matrix(modelo, X_test, y_test, normalize ='true',ax=ax,cmap="Reds")
    if save=="Y" or save=="y":
        plt.savefig("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/Stream_Confussion/CM_"+names[classifier_input]+"_"+str(number_of_features)+".jpg") 
    if show=="Y" or show=="y":
        plt.show()

confusion(Random,names[classifier_input],show_input,save_input)

#-------------------------------#
#   CLASSIFICATION AND SCORE    #
#-------------------------------#

X_GAIA=CLASSIFY[Features]
Estrellas_GAIA=CLASSIFY["Estrella"]
Y_GAIA=Random.predict(X_GAIA)

print("-------- Scores of final training test ---------")
print("Score %s : "%names[classifier_input])
print("Accuracy: %.5f" %Random.score(X_test,y_test))
print("F1 score: %.5f" %f1_score(Random.predict(X_test),y_test,average="weighted") )
print("Recall  : %.5f" %recall_score(Random.predict(X_test),y_test,average="weighted"))
print("%d \t %.5f \t %.5f \t %.5f" %(len(Features),Random.score(X_test,y_test),f1_score(Random.predict(X_test),y_test,average="weighted"),recall_score(Random.predict(X_test),y_test,average="weighted")))

ts_Score=datetime.datetime.now().timestamp()
print("Score done:"+str(ts_Score-ts1)+" Seconds\n\n")

print("--------      GAIA Results        ---------")
CLASS=pd.concat([Estrellas_GAIA,pd.Series(Y_GAIA)], axis=1)
CLASS.columns=["Estrella","TIPO"]
# CLASS.to_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/Clasification_FINAL.csv")
# print(CLASS.head())
COUNT=CLASS.groupby(["TIPO"]).count()
print(COUNT)

ts_GAIA=datetime.datetime.now().timestamp()
print("GAIA done:"+str(ts_GAIA-ts1)+" Seconds")

