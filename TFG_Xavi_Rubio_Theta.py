import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib import cm
#Unidades sistema internacional
#Definir posicions
hbarr=6.626*10**(-34)/(2*np.pi)
m0=9.11*10**(-31)
rmass = 0.067
Vbarrier = 0.25 #eV
V0 = Vbarrier * 1.6*10**(-19) # en Joules
q=-1.6*10**(-19) #Carga del electron en C

Lcell = 5.653 #Ang
L = 60 * Lcell # 60 celdas unidad forman nuestro material
Lua = L*10**(-10) #en metros
Lz = 20 *Lcell #L pozo en Ang (Valor encontrado en internet de pozo de AlGaAs/GaAs/AlGaAs)
Lzua = Lz*10**(-10) #en metros
dx = 12*10**-10 #Dist entre puntos en x en m
dy = 12*10**-10
dz = 12*10**-10
X = np.arange(-Lua/2,Lua/2,dx)
Y = np.arange(-Lua/2,Lua/2,dy)
Z = np.arange(-Lua/2,Lua/2,dz)


N=len(X) #Numero de puntos, hay que vigilar porque realmente se obtienen N^3 valores propios

##DEPENDENCIA CON THETA
print("Introduce el valor del campo magn (B) en T:")
B=float(input())
thetadeg = [0,15,30,45,60,75,90]
Theta = np.radians(thetadeg)

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
                Hx[i, j] = (hbarr**2 / (m0 * rmass * a**2))   # Diagonal
            elif abs(i - j) == 1:
                Hx[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hx

# Operador T en 1D, hacer call de la funcion Dy para dy
def Dy(a):
    Hy = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hy[i, j] = (hbarr**2 / (m0 * rmass * a**2))  # Diagonal
            elif abs(i - j) == 1:
                Hy[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hy
#Operador T en 1D + potencial pou quadrat, hacer call de la funcion Dz para dz
def Dz(a):
    Hz = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                Hz[i, j] = (hbarr**2 / (m0*rmass * a**2)) + potencial(Z[i])    #Diagonal
            elif abs(i - j) == 1:
                Hz[i, j] = -(hbarr**2 / (m0*2 * rmass * a**2)) #Diagonals superior e inferior
    return Hz
#Tensor T n^3 x n^3
T= np.kron(np.kron(Dx(dx),I),I) +np.kron(np.kron(I,Dy(dy)),I) + np.kron(np.kron(I,I),Dz(dz))
def VmagnzAZ(a,thet):
    AZ = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                AZ[i, j] = 1/(2 * rmass*m0)*((q*B*X[i]*np.sin(thet)/2)**2)  #Diagonal
            elif (i - j) == 1:
                AZ[i, j] = complex(0,-hbarr*q*B*X[i]*np.sin(thet)/(2*a* 2 * rmass*m0)) #Diagonal superior
            elif (i - j) == -1:
                AZ[i, j] = complex(0,hbarr*q*B*X[i]*np.sin(thet)/(2*a* 2 * rmass*m0)) #Diagonal inferior
    return AZ

def VmagnyAY(a,thet):
    AY = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                AY[i, j] = 1/(2 * rmass*m0)*((q*B*X[i]*np.cos(thet)/2)**2)  #Diagonal
            elif (i - j) == 1:
                AY[i, j] = complex(0,hbarr*q*B*X[i]*np.cos(thet)/(2*a* 2 * rmass*m0)) #Diagonals superior e inferior
            elif (i - j) == -1:
                AY[i, j] = complex(0,-hbarr*q*B*X[i]*np.cos(thet)/(2*a* 2 * rmass*m0)) #Diagonals superior e inferior
    return AY
def VmagnxAX(a,thet):
    AX = np.zeros((N, N),dtype=complex)
    for i in range(N):
        for j in range(N):
            if i == j:
                AX[i, j] = 1/(2 * rmass*m0)*((q*(B*Z[i]*np.sin(thet)-B*Y[i]*np.cos(thet))/2)**2)  #Diagonal
            elif (i - j) == 1:
                AX[i, j] = complex(0,-hbarr*q*B*(Z[i]*np.sin(thet)-Y[i]*np.cos(thet))/(2*a* 2 * rmass*m0)) #Diagonals superior e inferior
            elif (i - j) == -1:
                AX[i, j] = complex(0,hbarr*q*B*(Z[i]*np.sin(thet)-Y[i]*np.cos(thet))/(2*a* 2 * rmass*m0)) #Diagonals superior e inferior
    return AX

k0 = []
k1 = []
k2 = []
k3 = []
k4 = []
k5 = []
k6 = []
k7 = []
k8 = []
k9 = []
k10 =[]
x, y, z = np.meshgrid(X*10**9, Y*10**9, Z*10**9) #en nanometros
X_mesh, Y_mesh = np.meshgrid(X*10**9, Y*10**9)
for i in (Theta):
#Tensor d'energia potencial n^3 x n^3
    U=np.kron(np.kron(VmagnxAX(dx,i),I),I) + np.kron(np.kron(I,VmagnyAY(dy,i)),I) + np.kron(np.kron(I,I),VmagnzAZ(dz,i))
    Hamilt =T+U #Hamiltoniano
    eigenvalues , eigenvectors = eigsh(Hamilt, k=11,which="SM")
    id = np.argsort(eigenvalues)
    Val = eigenvalues[id]
    Vect = eigenvectors[:,id]
    eV=Val/(-q)
    k0.append(eV[0])
    k1.append(eV[1])
    k2.append(eV[2])
    k3.append(eV[3])
    k4.append(eV[4])
    k5.append(eV[5])
    k6.append(eV[6])
    k7.append(eV[7])
    k8.append(eV[8])
    k9.append(eV[9])
    k10.append(eV[10])

    #if i == Theta[0] or i == Theta[round(len(Theta)/6)] or i == Theta[round(2*len(Theta)/6)] or i == Theta[round(3*len(Theta)/6)] or i == Theta[round(4*len(Theta)/6)] or i == Theta[round(5*len(Theta)/6)] or i == Theta[-1]:
    def forma_eigvec(n):
        return Vect.T[n].reshape((N,N,N))
    for n in range (0,6):
            fig = plt.figure(1,figsize=(9,9))
            ax = fig.add_subplot(111, projection="3d")
            plot1 = ax.scatter3D(x, y, z, c=forma_eigvec(n),cmap=cm.seismic,s=1,alpha=0.6,antialiased=True)
            fig.colorbar(plot1, shrink=0.5, aspect=5,label="Real part of eigenfunction value")
        
            ax.set_xlabel(r"X (nm)")
            ax.set_ylabel(r"Y (nm)")
            ax.set_zlabel(r"Z (nm)")
            ax.set_title("Eigenfunction for {} state".format(n)+ " for Theta = {} degrees".format(round(np.degrees(i))) + " for B = {} T".format(B))
            plt.show()
            
            fig = plt.figure(1,figsize=(9,9))
            ax = fig.add_subplot(111, projection='3d')
            plot2 = ax.scatter3D(x, y, z, c=forma_eigvec(n)*np.conj(forma_eigvec(n)),cmap=cm.hot_r,s=1,alpha=0.6,antialiased=True)
            fig.colorbar(plot2, shrink=0.5, aspect=5,label="Probability density value")
            ax.set_xlabel(r"X (nm)")
            ax.set_ylabel(r"Y (nm)")
            ax.set_zlabel(r"Z (nm)")
            ax.set_title("Probability Density for {} state".format(n) + " for Theta = {} degrees".format(round(np.degrees(i))) + " for B = {} T".format(B))
            plt.show()

            plt.figure(figsize=(9, 9))
            plt.contourf(X_mesh, Y_mesh, forma_eigvec(n)[:,:,round(N/2)]*np.conj(forma_eigvec(n)[:,:,round(N/2)]), cmap=cm.hot)
            plt.colorbar(label="Probability Density value")
            plt.xlabel("X (nm)")
            plt.ylabel("Y (nm)")
            plt.title("Probability Density vs X,Y for Z=0 for {} state".format(n) + " for Theta = {} degrees".format(round(np.degrees(i))) + " for B = {} T".format(B))
            plt.show()
            
            plt.figure(5,figsize=(9, 9))
            plt.contourf(x[round(N/2), :, :], z[round(N/2), :, :],forma_eigvec(n)[round(N/2),:,:]*np.conj(forma_eigvec(n)[round(N/2),:,:]), cmap=cm.hot)
            plt.colorbar(label="Probability Density")
            plt.xlabel("X (nm)")
            plt.ylabel("Z (nm)")
            plt.title("Probability Density vs X,Z for Y=0 for {} state".format(n) + " for B = {} T".format(B))
            plt.show()

    plot4 = plt.figure(4)
    b = np.arange(0, len(eV),1)
    plt.scatter(b, eV, s=1444, marker="_", linewidth=2, zorder=3)
    plt.title("Plot of eigenvalues for Theta = {} degrees".format(round(np.degrees(i)))+ " for B = {} T".format(B))
    plt.xlabel('Energy level')
    plt.ylabel('Energy (eV)')
    c = ['$E_{}$'.format(i) for i in range(0,len(eV))]
    for i, txt in enumerate(c):
        plt.annotate(txt, (np.arange(0,len(eV),1)[i], eV[i]), ha="center")
    plt.show()

plt.plot(np.degrees(Theta),k0,label ="k=0",color="b")
plt.plot(np.degrees(Theta),k1,label ="k=1",color="r")
plt.plot(np.degrees(Theta),k2,label ="k=2",color="g")
plt.title ("E(eV) vs Theta(degrees) for B = {} T for the 3 lowest energy levels".format(B))
plt.xlabel("Theta(degrees)")
plt.ylabel("E(eV)")
plt.legend()
plt.show()



