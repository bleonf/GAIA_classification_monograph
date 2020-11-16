import Estadisticos
import numpy as np 
import matplotlib.pylab as plt 
import pandas as pd
import glob
import datetime
# import seaborn as sn
np.seterr(divide='ignore', invalid='ignore')

#Numero de estrellas para utilizar - Numero de Estrellas datos
# Nmin=5000
ts1=datetime.datetime.now().timestamp()

N1=287#287 #T2 287
N2=4626#1000 #CEP - 4626
N3=144#144 #ACEP - 144
N4=12500#1000 #RRLYRAE - 12527
N5=1043#1000 #ECL - 1043
N6=5000#1000 #LPV - 5000
N7=475#475 #BE - 475
N8=2788#1000 #DScuti - 2788
N9=0#6788 #GAIA - 6788
Numeros=[N1,N2,N3,N4,N5,N6,N7,N8,N9]
#Listas globs-n, con los titulos de los archivos literalmente
#de cada archivo en la lista de archivos genera un arreglo de numpy data#_k dividiendo los valores 

#Los mejores: Integral, Integral2, Proy, Proy2!!
# crea lista globs1  para t2cep_datos-287
if N1>0:
	globs1=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/T2CEP/T2CEP_I/*.dat')
	for k in range(N1):
		globals()["data1_"+str(k)]=np.genfromtxt(globs1[k],delimiter=' ',usecols=(0,1))
	print("T2 Listo")

# crea lista globs2  para cef_datos-4709
if N2>0:
	globs2=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/CEP/CEP_I/*.dat')
	for k in range(N2):
		globals()["data2_"+str(k)]=np.genfromtxt(globs2[k],delimiter=' ',usecols=(0,1))
	print("CEF Listo")

# crea lista globs3  para acef_datos-144
if N3>0:
	globs3=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/ACEP/ACEP_I/*.dat')
	for k in range(N3):
		globals()["data3_"+str(k)]=np.genfromtxt(globs3[k],delimiter=' ',usecols=(0,1))
	print("ACEF Listo")

# crea lista globs4  para rrly_datos-39220
if N4>0:
	globs4=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/RRLYRAE/RRLYR_I/*.dat')
	for k in range(N4):
		globals()["data4_"+str(k)]=np.genfromtxt(globs4[k],delimiter=' ',usecols=(0,1))
	print("RRlyr Listo")

# crea lista globs1  para ecl_datos-1145
if N5>0:
	globs5=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/ECL/ECL_I/*.dat')
	for k in range(N5):
			globals()["data5_"+str(k)]=np.genfromtxt(globs5[k],delimiter=' ',usecols=(0,1))
	print("ECL Listo")

# crea lista globs6  para LPV-5000
if N6>0:
	globs6=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/LPV/LPV_I/*dat')
	for k in range(N6):
		globals()["data6_"+str(k)]=np.genfromtxt(globs6[k],delimiter=' ',usecols=(0,1))
	print("LPV Listo")

	# crea lista globs7  para Be-5000
if N7>0:
	globs7=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/BE/becand2/I/*.dat')
	# print(len(globs7))
	for k in range(N7):
		globals()["data7_"+str(k)]=np.genfromtxt(globs7[k],delimiter=' ',usecols=(0,1))
	print("BE Listo")
	# crea lista globs7  para Dscuti 2788
if N8>0:
	globs8=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Entrenamiento/DScuti/I/*.dat')
	# print(len(globs8))
	for k in range(N8):
		globals()["data8_"+str(k)]=np.genfromtxt(globs8[k],delimiter=' ',usecols=(0,1))
	print("DScuti Listo")

if N9>0:
	Datos_GAIA=glob.glob('/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Clasificar/GAIA_I/*.dat')
	print(len(Datos_GAIA))
	for k in range(N9):
		globals()["data9_"+str(k)]=np.genfromtxt(Datos_GAIA[k],delimiter=' ',usecols=(0,1))
	print("Gaia Listo")
	with open('/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/GAIA_names.txt', 'w') as f:
		for item in Datos_GAIA:
			f.write("%s \n" %item.replace("/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Clasificar/GAIA_I/",""))
ts_datos=datetime.datetime.now().timestamp()
print("Datos listos:"+str(ts_datos-ts1)+" Segundos")
 
nombres=["ABBE","MAD","OCTILa","OCTILb","SKEW","MEDIAN","PITAR","RULD","TFACT2","TFACT","DIFER","HISTHIST","INTEGRAL","INTEGRAL2","MADMAD","DERPROM","DIFDER","PROY","PROY2","MEGAMAD","TANGENTE","DERMED","CONTPOS"]
# Letras
a,b,c,d,e,f,g,h,ii,j,Kk,l,m,n,o,p,q,r,s,t,u,v,w=nombres
A,B,C,D,E,F,G,H,I,J,Kkk,L,M,N,O,P,Q,R,S,T,U,V,W,=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
estrella=[]
TIPO=[]


metodos={"A":Estadisticos.ABBE,"B":Estadisticos.MAD,"C":Estadisticos.OCTILa,"D":Estadisticos.OCTILb,"E":Estadisticos.SKEW,"F":Estadisticos.Mediana,"G":Estadisticos.PITAR,"H":Estadisticos.RULD,"I":Estadisticos.TFACT2,"J":Estadisticos.TFACT,"K":Estadisticos.DIFER,"M":Estadisticos.INTEGRAL,"N":Estadisticos.INTEGRAL2,"O":Estadisticos.MADMAD,"P":Estadisticos.DERPROM,"Q":Estadisticos.DIFDER,"R":Estadisticos.PROY,"S":Estadisticos.PROY2,"T":Estadisticos.MEGAMAD,"U":Estadisticos.TANGENTE,"V":Estadisticos.DERMED,"W":Estadisticos.CONTPOS}

Tipos={"T2":N1,"CEP":N2,"ACEP":N3,"RRLYRAE":N4,"ECL":N5,"LPV":N6,"Be":N7,"DSCUTI":N8,"GAIA":N9}

listas={"A":A,"B":B,"C":C,"D":D,"E":E,"F":F,"G":G,"H":H,"I":I,"J":J,"K":Kkk,"M":M,"N":N,"O":O,"P":P,"Q":Q,"R":R,"S":S,"T":T,"U":U,"V":V,"W":W}

def CREATOR(func,Z):
	Z+=[func(globals()["data1_"+str(i)]) for i in range(N1)]
	Z+=[func(globals()["data2_"+str(i)]) for i in range(N2)]
	Z+=[func(globals()["data3_"+str(i)]) for i in range(N3)]
	Z+=[func(globals()["data4_"+str(i)]) for i in range(N4)]
	Z+=[func(globals()["data5_"+str(i)]) for i in range(N5)]
	Z+=[func(globals()["data6_"+str(i)]) for i in range(N6)]
	Z+=[func(globals()["data7_"+str(i)]) for i in range(N7)]
	Z+=[func(globals()["data8_"+str(i)]) for i in range(N8)]
	Z+=[func(globals()["data9_"+str(i)]) for i in range(N9)]
	return "Listo"

estrella=[] 
for k in range(9):
	if Numeros[k]>0:
		if k==8:
			estrella+=[item.replace("/home/bleon/Documents/TESIS_FILES/Datos_Nuevos/Clasificar/GAIA_I/","") for item in Datos_GAIA]
		else:
			estrella+=["data"+str(k+1)+"_"+str(i) for i in range(Numeros[k])] #Lista de estrellas usadas

for tipo in Tipos.keys():
	TIPO+=[tipo for i in range(Tipos[tipo])] #Lista de tipos 

for indice in metodos.keys():
	CREATOR(metodos[indice],listas[indice]) #Creator de datos
ts_creator=datetime.datetime.now().timestamp()
print("Creator listo:"+str(ts_creator-ts1)+" Segundos")
 

data={'Estrella':estrella,'TIPO':TIPO}


data = {'Estrella':estrella,a:A,b:B,c:C,d:D,e:E,f:F,g:G,h:H,ii:I,j:J,Kk:Kkk,m:M,n:N,o:O,p:P,q:Q,r:R,s:S,t:T,u:U,v:V,w:W,'TIPO':TIPO,}

datos=pd.DataFrame(data,columns=["Estrella",a,b,c,d,e,f,g,h,ii,j,Kk,m,n,o,p,q,r,s,t,u,v,w,"TIPO"])

datosCorr=datos.corr()
df=datosCorr
datos.to_csv("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/datos_Final_entrenamiento.csv")

f= plt.figure(figsize=(8, 8))
plt.matshow(datosCorr, fignum=f.number,cmap="RdBu")
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
plt.plot([5.5,5.5],[-0.5,21.5],c="red")
plt.plot([-0.5,21.5],[5.5,5.5],c="red")
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

ts_cor=datetime.datetime.now().timestamp()
print("Correlacion listo:"+str(ts_cor-ts1)+"Segundos")
# plt.savefig("/home/bleon/Documents/TESIS_FILES/Codigos/DATOS/Matriz_Correlacion.png")
plt.show()




