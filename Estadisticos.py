import numpy as np
np.seterr(divide='ignore', invalid='ignore')
#INTENSIDAD= datos[:,1]
#TIEMPO= datos[:,0]
#Funciones en Estadisticos.py
#DIFDER
#PROY Bien
#PROY2
#HISTHIST
#DERPROM
#INTEGRAL2
#INTEGRAL
#DIFER
#TFACT2	FOR
#TFACT	FOR
#RULD	
#RULD2
#PITAR
#MADMAD Muy bien-Completamente relacionado con MAD
#ABBE	Muy bien	 
#MAD	Muy bien	 
#SKEW	Muy bien	 
#OCTILb	Muy bien	 
#OCTILa	Muy bien	 
#Mediana	Muy bien	 



# def PREPOS(datos):
# 	datosmag=datos[:,1]-np.median(datos[:,1])
# 	datostemp=datos[:,0]-datos[0,0]
# 	dev=(np.max(datosmag)/np.median(datosmag))/15
# 	datsM=datosmag[0:-2]-datosmag[1:-1]#Diferencias en magnitud
# 	datsT=datostemp[0:-2]-datostemp[1:-1]#diferencias en tiempo

# def FANTASMA(datos):
	# med=np.median(datos)
	# amp=((np.max(datos)-med)+(np.min(datos)-med))/2.0
	# 
def DISTBack(datos):
	"""Promedio de la distancia entre un punto y la extrapolación lineal de los siguientes dos puntos, medida de curvatura 2"""
	return 0

def DISTFORWARD(datos):
	"""Promedio de la distancia entre el punto siguiente y la extrapolación lineal de dos puntos, medida de curvatura"""
	return 0

def CONTPOS(datos):
	""" Proporcion de derivadas ascendientes contra derivadas descendientes"""
	der=((datos[0:-2,1]-datos[1:-1,1])/(datos[0:-2,0]-datos[1:-1,0]))
	a=0
	b=0
	for valor in der:
		if valor > 0:
			a+=valor
		else: 
			b-=valor
	return a/b	
	
	
def DERMED(datos):
	"""Mediana de las derivadas"""
	datos=np.array(datos)
	der=((datos[0:-2,1]-datos[1:-1,1])/(datos[0:-2,0]-datos[1:-1,0]))
	return np.median(der)

def TANGENTE(datos):
	"""valor de la mediana de las proyecciones de la linea tangente sobre la linea vertical a 100 dias de distancia"""
	datos=np.array(datos) 
	der=np.array((datos[0:-2,1]-datos[1:-1,1])/(datos[0:-2,0]-datos[1:-1,0]))
	# datos_mag=np.array(datos[0:-2,1])#-np.mean(datos[:,1])
	proy_100=(100.0*der)#+datos_mag
	proy_100_condicional=np.where(proy_100>0,1,-1)
	return np.abs(np.sum(proy_100_condicional)/len(proy_100_condicional))


def PROY2(datos): #
	"""Distancia entre un punto y la linea que conecta sus 2 vecinos"""
	datos=np.array(datos) 
	der=((datos[2:,1]-datos[:-2,1])/(datos[2:,0]-datos[:-2,0]))
	azul=np.sum(np.abs((der[:]*datos[1:-1,0])+(datos[2:,1]-(der[:])*datos[2:,0])-datos[1:-1,1]))
	return azul/(len(der))

def PROY(datos):
	"""Proyeccion lineal sin ajuste de intercepto, lineas parten del origen"""
	datos=np.array(datos) 
	der=((datos[2:,1]-datos[:-2,1])/(datos[2:,0]-datos[:-2,0]))
	azul=np.sum((der[:]*(datos[1:-1,0]))-datos[1:-1,1])
	return azul/(len(der)*len(datos[:,0]))

def DIFDER(datos): #Diferencia de derivadas (derivada 0-derivada2/punto 1)
	"""Sumatoria de cosiente entre la diferencia de derivadas y diferencia de magnitudes"""
	datos=np.array(datos) 
	der=np.divide((datos[0:-2,1]-datos[1:-1,1]),(datos[0:-2,0]-datos[1:-1,0])) #Derivadas
	azul=np.sum(np.divide((der[0:-3]-der[2:-1]),(datos[1:-4,0]-datos[2:-3,0]))) #
	return azul/len(der)

def DERPROM(datos): 
	"""Promedio de las derivadasss"""
	datos=np.array(datos) 
	suma=np.sum(np.divide((datos[0:-2,1]-datos[1:-1,1]),(datos[0:-2,0]-datos[1:-1,0])))
	return suma

def INTEGRAL2(datos): 
	"""Sumatoria de Riemman normal""" #Solo suma
	datos=np.array(datos) 
	datosmag=datos[:,1]-np.median(datos[:,1])
	datostemp=datos[:,0]-datos[0,0] #Datos temporales menos primer dato
	suma=np.sum(np.abs(datosmag[0:-1]*(datostemp[0:-1]-datostemp[1:])))
	return suma/(len(datostemp)-1)

def INTEGRAL(datos): #Sumatoria de Riemman #Suma y resta
	datos=np.array(datos) 
	datosmag=datos[:,1]-np.median(datos[:,1])
	datostemp=datos[:,0]-datos[0,0]
	suma=np.sum((datosmag[0:-1]*(datostemp[1:]-datostemp[:-1])))
	return suma

def HISTHIST(datos):#retorna regresion lineal de la distribucion de derivadas
	datos=np.array(datos) 
	deriv=np.divide((datos[1:-1,1]-datos[0:-2,1]),(datos[1:-1,0]-datos[0:-2,0]))
	y,x=np.histogram((deriv))
	a=np.polyfit(x[1:],y,1)
	return (a[0])

def DIFER(datos): #Promedio de X-mediana / T-To
	datos=np.array(datos) 
	medmag=np.median(datos[:,1])
	primtemp=datos[0,0]
	deriv=np.divide((datos[1:,1]-medmag),(datos[1:,0]-primtemp))
	tot=np.sum(deriv)/float(len(datos[:,0])-1)
	return tot

def TFACT(datos): #dat0s[:,1] #derivada al cuadrado sumatoria	
	datos=np.array(datos) 	
	tfact_datos=np.sum(np.divide(((datos[:-1,1]-datos[1:,1])**2),((datos[:-1,0]-datos[1:,0])**2)))
	if len(datos[1:,0])>0:
			tfact_datos=tfact_datos/len(datos[1:-1,0])
	return tfact_datos

def TFACT2(datos): #diferencia pasada al cuadrado sobre diferencia siguiente al cuadrado en magnitud
	datos=np.array(datos) 
	tfact_datos2=0
	for i in range(len(datos[1:-3,0])):
		tfact_datos2+=(((datos[i,1]-datos[i-1,1]))/((datos[i+2,0]-datos[i+1,0]+0.0000001)))
	tfact_datos2/=len(datos[1:-3,0])
	return tfact_datos2

def RULD(datos): #dat0s[:,1] #SI dato es mayor al anterior, suma diferencia temporal, menos resta
	datos=np.array(datos) 
	"""Si dato asciende se suma la diferencia temporal, si desciende se resta. Todo se divide por la longitud de los datos """
	ruld_datos=0
	for i in range(len(datos[:-1,0])):
		if (datos[i,1]>datos[i+1,1]):
			ruld_datos=ruld_datos-(datos[i,0]-datos[i+1,0])
		if (datos[i+1,1]<datos[i+1,1]):	
			ruld_datos=ruld_datos+(datos[i,0]-datos[i+1,0])
	ruld_datos=ruld_datos/float(len(datos[1:,0]))
	return ruld_datos

def RULD2(datos): #dat0s[:,1] #SI dato es mayor al anterior, suma diferencia temporal, menos resta.
	datos=np.array(datos) 
	"""Si dato asciende se suma la diferencia temporal, si desciende se resta. Todo se divide por la diferencia temporal entre el primer y el utlimo dato """ #Balance entre cuanto tiempo la serie de tiempo esta subiendo contra cuanto esta bajando
	ruld2_datos=0
	for i in range(len(datos[:-1,0])):
		if (datos[i,1]>datos[i+1,1]):
			ruld2_datos-=(datos[i,0]-datos[i+1,0])
		if (datos[i+1,1]<datos[i+1,1]):	
			ruld2_datos+=(datos[i,0]-datos[i+1,0])
	ruld2_datos/=(datos[-1,0]-datos[0,0])
	return ruld2_datos

def PITAR(datos): #dat0s[:,1] #distancia topologica 
	datos=np.array(datos) 
	pitar_datos=0
	pitar_datos=np.sum(np.sqrt(np.square((datos[1:-1,0]-datos[0:-2,0]))+np.square((datos[1:-1,1]-datos[0:-2,1]))))
	pitar_datos/=float(len(datos[1:-1]))
	return pitar_datos

def SKEW(datos): #Medida oblicuidad o cesgo
	datos=np.array(datos) 
	skew_datos=0
	Q50=np.median(datos[:,1])
	datos1=np.sort(datos[:,1])
	Q875=datos1[int(0.875*len(datos[:,1]))]
	Q125=datos1[int(0.125*len(datos[:,1]))]
	skew_datos=((Q875-Q50)-(Q50-Q125))/(Q875-Q125)
	return skew_datos

def MEGAMAD(datos):
	datos=np.array(datos) 
	MAD_dat=np.median(abs((datos[:,1]-np.median(datos[:,1]))))
	MAD_temp=np.median(abs((datos[:,0]-np.median(datos[:,0]))))
	megamad=MAD_dat/MAD_temp
	return megamad

def MAD(datos): #Median Absolute deviation
	datos=np.array(datos) 
	MAD_datos=np.median(abs((datos[:,1]-np.median(datos[:,1]))))
	return MAD_datos

def MADMAD(datos): #Median Absolute deviation**2
	datos=np.array(datos) 
	MADMAD_datos=np.median(abs((datos[:,1]-np.median(abs((datos[:,1]-np.median(datos[:,1])))))))
	return MADMAD_datos

def OCTILb(datos): #peso de octil por derecha
	datos=np.array(datos) 
	octilb_datos=0
	datos1=np.sort(datos[:,1])
	Q625=datos1[int(0.625*len(datos[:,1]))]
	Q75=datos1[int(0.75*len(datos[:,1]))]
	Q875=datos1[int(0.875*len(datos[:,1]))]
	octilb_datos=((Q875-Q75)-(Q75-Q625))/(Q875-Q625)
	return octilb_datos

def OCTILa(datos): #peso de octil por izquierda
	datos=np.array(datos) 
	octila_datos=0
	datos1=np.sort(datos[:,1])
	Q125=datos1[int(0.125*len(datos[:,1]))]
	Q25=datos1[int(0.25*len(datos[:,1]))]
	Q375=datos1[int(0.375*len(datos[:,1]))]
	octila_datos=((Q375-Q25)-(Q25-Q125))/(Q375-Q125)
	return octila_datos

def Mediana(datos): #mediana
	datos=np.array(datos) 
	mediana_datos=np.median(datos[:,1])
	return mediana_datos

def ABBE(datos):#valor de Abbe
	try:
		datos=np.array(datos) 
		suma1=np.sum(np.square(datos[1:-1,1]-datos[0:-2,1]))
		suma2=np.sum(np.square(datos[:,1]-np.average(datos[:,1])))
		leng=float(len(datos[:,1]))
		nivel=leng/(2*(leng-1))	
		return ((suma1*nivel)/suma2)
	except:
		return(0)

def CANTIDAD(datos):
	a=datos.shape[0]
	return(a)
