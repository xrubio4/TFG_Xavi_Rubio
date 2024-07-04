import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

#Unidades sistema internacional
#Definir posicions
hbarr=6.626*10**(-34)/(2*np.pi)
m0=9.11*10**(-31)
rmass = 0.067
Vbarrier = 0.25 #eV
V0 = Vbarrier * 1.6*10**(-19) # en Joules
q=-1.6*10**(-19) #Carga del electron en C

Lcell = 5.653 #Ang
L = 60 * Lcell # 100 celdas unidad forman nuestro material
Lua = L*10**(-10) #en metros
Lz = 20 *Lcell #L pozo en Ang 
Lzua = Lz*10**(-10) #en metros
dx = 15*10**-10 #Dist entre puntos en x en m
dy = 15*10**-10
dz = 15*10**-10
X = np.arange(-Lua/2,Lua/2,dx)
Y = np.arange(-Lua/2,Lua/2,dy)
Z = np.arange(-Lua/2,Lua/2,dz)

N=len(X) #Numero de puntos, hay que vigilar porque realmente se obtienen N^3 valores propios

rangB = [0,50,100,150,200,300,500]

I=np.identity(N) #Matriz identidad
#Operadores
#Potencial pozo cuadrado en Z
def potencial(z):
    if (abs(z)<(Lzua/2)):
        return 0
    else:
        return V0
# Operador T en 1D, hacer call de la funcion Dx para dx
def Dx(a):
    Hx = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hx[i, j] = (hbarr**2 / (m0 * rmass * a**2))    # Diagonal
            elif abs(i - j) == 1:
                Hx[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonales superior e inferior
    return Hx

# Operador T en 1D, hacer call de la funcion Dy para dy
def Dy(a):
    Hy = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hy[i, j] = (hbarr**2 / (m0 * rmass * a**2))   # Diagonal
            elif abs(i - j) == 1:
                Hy[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonales superior e inferior
    return Hy
#Operador T en 1D + potencial pou quadrat, hacer call de la funcion Dz para dz
def Dz(a):
    Hz = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hz[i, j] = (hbarr**2 / (m0*rmass * a**2)) +potencial(Z[i])     #Diagonal
            elif abs(i - j) == 1:
                Hz[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonales superior e inferior
    return Hz

#Tensor T n^3 x n^3
T= np.kron(np.kron(Dx(dx),I),I) +np.kron(np.kron(I,Dy(dy)),I) + np.kron(np.kron(I,I),Dz(dz)) 
def VmagnAX(a,B):
    V = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                V[i, j] = ((q*B*Y[i]/2)**2)/(2 * rmass*m0)  #Diagonal
            elif (i - j) == 1:
                V[i, j] = complex(0,-hbarr*q*B*Y[i]/(2*a* 2 * rmass*m0)) #Diagonal superior
            elif (i - j) == -1:
                V[i, j] = complex(0,hbarr*q*B*Y[i]/(2*a* 2 * rmass*m0)) #Diagonal inferior
    return V
def VmagnAY(a,B):
    V = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                V[i, j] = ((q*B*X[i]/2)**2)/(2 * rmass*m0)  #Diagonal
            elif (i - j) == 1:
                V[i, j] = complex(0,-hbarr*q*B*X[i]/(2*a* 2 * rmass*m0)) #Diagonal superior
            elif (i - j) == -1:
                V[i, j] = complex(0,hbarr*q*B*X[i]/(2*a* 2 * rmass*m0)) #Diagonal inferior
    return V

# Crear una función para calcular la densidad de estados
def calcular_densidad_de_estados(energias, ancho_de_bandas=0.1):
    densidad = gaussian_kde(energias, bw_method=ancho_de_bandas)
    energia_min = min(energias)
    energia_max = max(energias)
    print("Energía màxima:")
    print(energia_max)
    energia_range = np.linspace(energia_min,energia_max , len(eV))
    densidad_de_estados = densidad(energia_range)
    return energia_range, densidad_de_estados

for i in (rangB):
#Tensor d'energia potencial n^3 x n^3
    U=np.kron(np.kron(VmagnAX(dx,i),I),I) + np.kron(np.kron(I,VmagnAY(dy,i)),I)
    Hamilt =T+U #Hamiltoniano
    eigenvalues , eigenvectors = eigsh(Hamilt, k=11,which="SM")
    id = np.argsort(eigenvalues)
    Val = eigenvalues[id]
    Vect = eigenvectors[:,id]
    eV=Val/(-q)
    # Calcular la densidad de estados
    energia_range, densidad_de_estados = calcular_densidad_de_estados(eV)

    energia_range_normalizada = energia_range / max(eV)
 
    # Graficar la densidad de estados vs. energía
    plt.figure(figsize=(10, 6))
    plt.plot(energia_range_normalizada, densidad_de_estados)
    plt.title("Density of states vs Energy for B = {} T ".format(i) +"for Emax = {} eV".format(round(max(eV),2)))
    plt.xlabel("Energy/Emax")
    plt.ylabel("Density of states")
    plt.show()

