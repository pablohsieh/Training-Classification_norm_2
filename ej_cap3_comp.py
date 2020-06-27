import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
from scipy.integrate import quad

# Muestras
# s_w_j[i] es el vector de muestras del feature x_i de la clase j, i={0,2}
s_w_1 = np.array([ [-0.42,-0.2,1.3,0.39,-1.6,-0.029,-0.23,0.27,-1.9,0.87],
                   [-0.087,-3.3,-0.32,0.71,-5.3,0.89,1.9,-0.3,0.76,-1.0],
                   [0.58,-3.4,-1.7,0.23,-0.15,-4.7,2.2,-0.87,-2.1,-2.6] ])

#s_w_2=np.array([[-0.4,-0.31,0.38,-0.15,-0.35,0.17,-0.011,-0.27,-0.065,-0.12],
#                  [0.58,0.27,0.55,0.53,0.47,0.69,0.55,0.61,0.49,0.054],
#                  [0.089,-0.04,-0.035,0.011,0.034,0.1,-0.18,0.12,0.0012,
#                   -0.063]])

s_w_3 = np.array([ [0.83,1.1,-0.44,0.047,0.28,-0.39,0.34,-0.3,1.1,0.18],
                   [1.6,1.6,-0.41,-0.45,0.35,-0.48,-0.079,-0.22, 1.2, -0.11],
                   [-0.014,0.48,0.32,1.4,3.1,0.11,0.14,2.2,-0.46,-0.49] ])

p_w1 = p_w3 = 0.5 #Probabilidades a priori de las clases, se las va a sumir =
#p_w2 = 0.5

#ej_x_w_1 = np.array([[-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.51,-2.25,5.56,1.03],
#               [-8.12,-3.48,-5.52,-3.78,0.63,3.29,2.09,-2.13,2.86,-3.33],
#               [-3.68,-3.54,1.66,-4.11,7.39,2.08,-2.59,-6.94,-2.26,4.33]])

#ej_x_w_3 = np.array([[5.35,5.12,-1.34,4.48,7.11,7.17,5.75,0.77,0.90,3.52],
#               [2.26,3.22,-5.31,3.42,2.39,4.33,3.97,0.27,-0.43,-0.36],
#               [8.13,-2.66,-9.87,5.19,9.21,-0.98,6.65,2.41,-8.71,6.43]])

# Se puede ver como estan distribuidas las muestras y estimar a ojo
#fig_1 = plt.figure(1)
#plt.plot(s_w_1[0][:],s_w_1[1][:],'o',linewidth=1,label='w_1')#x1 vs x2 de w_1
#plt.plot(s_w_3[0][:],s_w_3[1][:],'o',linewidth=1,label='w_3')#x1 vs x2 de w_3
#plt.xlabel('Feature x_1')
#plt.ylabel('Feature x_2')
#plt.legend()
#plt.show()

#fig_2 = plt.figure(2)
#ax = fig_2.add_subplot(111, projection='3d')
#w1_x_1 =s_w_1[0][:]
#w1_x_2 =s_w_1[1][:]
#w1_x_3 =s_w_1[2][:]
#w3_x_1 =s_w_3[0][:]
#w3_x_2 =s_w_3[1][:]
#w3_x_3 =s_w_3[2][:]
#ax.scatter(w1_x_1, w1_x_2, w1_x_3, c='r', marker='o',label='w_1')
#ax.scatter(w3_x_1, w3_x_2, w3_x_3, c='b', marker='o',label='w_3')
#ax.set_xlabel('Feature x_1')
#ax.set_ylabel('Feature x_2')
#ax.set_zlabel('Feature x_3')
#plt.legend()
#plt.show()

# Estimador de max verosimilitud gaussiano -----------------------------------
def MLE_theta(muestra,cant_features):
#Obtengo el vector de medias (o la media si cant_features=1)
    mu = np.zeros(cant_features)
    for i in range(cant_features):
        mu[i] = np.mean(muestra[i])
#Obtengo la matriz de covarianza (o la varianza si cant_features=1)
    sigma = np.zeros((cant_features,cant_features))
    for i in range(cant_features):
        for j in range(cant_features):    
            sigma[i][j] = np.dot((muestra[:][i]-mu[i]),(muestra[:][j]-mu[j]))
    sigma = sigma / 10
    return mu, sigma

# Gaussiana ------------------------------------------------------------------
def gaussiana(x,mu,sigma): 
#Es la pdf gaussiana
#Para el caso del error se usa mu=0, sigma=1
    return (np.exp( (-(x-mu)**2)/(2*sigma**2)))*(1/(np.sqrt(2*np.pi)*sigma))

def muestras_gauss(n,m,s):
# Obtencion de un array de n muestras gaussianas de media y varianza aleatoria
# mu = [-m;m] y sigma=[0;s]
    muestra = np.zeros(n)
    for i in range(n):
        mu = np.random.randint(m) * np.random.uniform(0,1) 
        mu = (-1)**(np.random.randint(10000)) * mu #aleatorio en signo
        sigma = np.random.randint(s) * np.random.uniform(0,1)
        #print(mu,sigma)
        muestra[i] = np.random.normal(mu,sigma,1)
    return muestra

# Error ----------------------------------------------------------------------
def mahalanobis_distance(d,x,y,cov_xy): #distancia de mahalanobis entre x e y
    a = x - y
    if d ==1: #caso escalar
        b = 1/cov_xy
    else: #caso de dimension > 1
        b = np.linalg.inv(cov_xy)
    r = np.dot(a,np.dot(a,b))
    return np.sqrt(r)

def p_error(lim_inf): # Probabilidad de error integrando pdf gaussiana
# el lim inferior es la dist mahalanobis/2
    return quad(gaussiana,lim_inf,np.inf,args=(0,1))
    
# CLASIFICADOR DICOTOMICO ----------------------------------------------------
def discriminante(d,muestra,mu,sigma,pap):
# pap:probabilidad a priori de las clase
# d:cant de features a analizar
    a = muestra - mu    
    if d == 1: #caso escalar
        sigma_inv = 1/sigma
        det_sigma = sigma
    else: #caso de dimension > 1
        sigma_inv = np.linalg.inv(sigma)    
        det_sigma = np.linalg.det(sigma)
    aux = -0.5*np.dot(a,np.dot(a,sigma_inv))
    return aux - (d/2)*np.log(2*np.pi) - 0.5*np.log(det_sigma) + np.log(pap)
    
def es_clase_a(d,muestra,mu_a,sigma_a,pap_a,mu_b,sigma_b,pap_b):
# Se imprime de que clase es la muestra y cual es el error de la clasif
    discr_a = discriminante(d,muestra,mu_a,sigma_a,pap_a)
    discr_b = discriminante(d,muestra,mu_b,sigma_b,pap_b)
    g = discr_a - discr_b 
    if g > 0: # es de clase a
        error_clasif = p_error(mahalanobis_distance(d,muestra,mu_a,sigma_a))        
        error_scientific = "{:.4e}".format(error_clasif[0]) #para imprimir
        print ('    La muestra',muestra,
               'es de clase 1. Error de clasificación =',
               error_scientific,file=f)
    else: # es de clase b
        error_clasif = p_error(mahalanobis_distance(d,muestra,mu_b,sigma_b))
        error_scientific = "{:.4e}".format(error_clasif[0]) #para imprimir
        print ('    La muestra',muestra,
               'es de clase 3. Error de clasificación =',
               error_scientific,file=f)

# Redefinicion de mu, sigma, muestras dependiente de features-----------------
POS_X1 = 0 #para facilitar la escritura de los features
POS_X2 = 1
POS_X3 = 2
def cuales_features(x1,x2,x3,media,sigma,sample): 
#Fun para armar los vectores de media y matriz de cov dependiendo los features
    if x1 == 0:
        if x2 == 0:
            if x3 == 1:
                #x1=0 ^ x2=0 ^ x3=1
                return [ media[POS_X3],
                         sigma[POS_X3][POS_X3],
                         np.array([sample[2]]) ]
        elif x2 == 1:
            if x3 == 0:
                #x1=0 ^ x2=1 ^ x3=0
                return [ media[POS_X2],
                         sigma[POS_X2][POS_X2],
                         np.array([sample[1]]) ]
            elif x3 == 1:
                #x1=0 ^ x2=1 ^ x3=1
                return [ np.array([ media[POS_X2],media[POS_X3] ]),
                         np.array([ [ sigma[POS_X2][POS_X2],
                                     sigma[POS_X2][POS_X3] ],
                           [ sigma[POS_X3][POS_X2],sigma[POS_X3][POS_X3] ] ]),
                         np.array([sample[1],sample[2]]) ]
    elif x1 == 1:
        if x2 == 0:
            if x3 == 0:
                #x1=1 ^ x2=0 ^ x3=0
                return [ media[POS_X1],
                         sigma[POS_X1][POS_X1],
                         np.array([sample[0]]) ]
            elif x3 ==1:
                #x1=1 ^ x2=0 ^ x3=1
                return [ np.array([ media[POS_X1],media[POS_X3] ]),
                         np.array([ [ sigma[POS_X1][POS_X1],
                                     sigma[POS_X1][POS_X3] ],
                           [ sigma[POS_X3][POS_X1],sigma[POS_X3][POS_X3] ] ]),
                         np.array([sample[0],sample[2]]) ]
        elif x2 == 1:
            if x3 == 0:
                #x1=1 ^ x2=1 ^ x3=0
                return [ np.array([ media[POS_X1],media[POS_X2] ]),
                         np.array([ [ sigma[POS_X1][POS_X1],
                                     sigma[POS_X1][POS_X2] ],
                           [ sigma[POS_X2][POS_X1],sigma[POS_X2][POS_X2] ] ]),
                         np.array([sample[0],sample[1]]) ]
            elif x3 == 1:
                #x1=1 ^ x2=1 ^ x3=1
                return [ media,
                         sigma,
                         sample]        

def nuevas_mu_sigma_sample(mu,sigma,sample,comb):
# Dependiendo la combinacion de comb=[x1,x2,x3] se arma el nuevo 
#   vector de medias, matriz de covarianza y vector de muestras
    mu_out, sigma_out, sample_out = cuales_features(comb[0],comb[1],comb[2]
                                                    ,mu,sigma,sample)
    return [mu_out,sigma_out,sample_out]

# Realizacion del ejercicio --------------------------------------------------
def clasif(comb,sample,mu_i,sigma_i,pap_i,mu_j,sigma_j,pap_j):
# Realizacion del ejercicio
    print('Análisis con features x1=',comb[0],', x2=',comb[1],
          'y x3=',comb[2],'        ----------',np.sum(comb),
          'features ----------',file=f)
    mu_a , sigma_a, muestra = nuevas_mu_sigma_sample(mu_i,sigma_i,sample,comb)
    mu_b , sigma_b, muestra = nuevas_mu_sigma_sample(mu_j,sigma_j,sample,comb)
    es_clase_a(np.sum(comb),muestra,mu_a,sigma_a,pap_i,mu_b,sigma_b,pap_j)
    
    
# main ----------------------------------------------------------------------

n_features = 3  #cantidad de features a utilizar 1-3
m=10 # media entre -10 y 10
s=5 # varianza entre 0 y 5

# Obtengo la media y varianza MLE a partir de las muestras de entrenamiento
media_1 , sigma_1 = MLE_theta(s_w_1,n_features)
media_3 , sigma_3 = MLE_theta(s_w_3,n_features)

f = open("ej_cap3_comp_result.txt", "w") # Voy a imprimirlo en un archivo

for i in range(5):
    print('Realización',i+1,file=f)
    sample_a_clasificar = muestras_gauss(3,m,s)
    
    # 3 features   
    comb = np.array([1,1,1])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    
    # 2 features    
    comb = np.array([1,0,1])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    comb = np.array([1,1,0])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    comb = np.array([0,1,1])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    
    # 1 feature
    comb = np.array([1,0,0])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    comb = np.array([0,1,0])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    comb = np.array([0,0,1])
    clasif(comb,sample_a_clasificar,media_1,sigma_1,p_w1,media_3,sigma_3,p_w3)
    
    print('\n',file=f)

f.close()

#samp = muestras_gauss(3)
#print('\n')
#print('dimension 3 r=',mahalanobis_distance(3,samp[0:3],media_1,sigma_1))
#print('dimension 2 r=',mahalanobis_distance(2,samp[0:2],media_12,sigma_12))
#print('dimension 1 r=',mahalanobis_distance(1,samp[0:1],media_11,sigma_11)) 
 