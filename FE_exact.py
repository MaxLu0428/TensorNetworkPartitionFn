from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pandas as pd
import os




XX =[]
YY =[]



step= 40
deltT = (4.0/step)

T_list  = [  deltT*i  for i in range(1,step+1) ]


FE_D = dict()


ti =0; FE_list = np.zeros(step)

for T in T_list:


    beta = 1./T  #-0.25*np.log(za)

    XX.append(T)

    kk = 2*np.sinh(2*beta)/(np.cosh(2*beta)**2)

    dth = 0.001
    thelist = np.arange(0,pi,dth)


    val = 0.0
    for ii in thelist:
        val = val + np.log(0.5*(1.+ np.sqrt(1-(kk**2)*(np.sin(ii))**2) ))*dth

    val = val/2/pi
    FF =np.log(2* np.cosh(2*beta) * np.exp(val))
    FF = -FF*T

    YY.append(FF)
    
    FE_list[ti] = FF;   ti+=1


print (len(FE_list))
print (len(T_list))
FE_D[0] = FE_list





#directory = './Ising'  
#os.makedirs(directory)
#filename = directory +'/FE_D' + str(dcut) + '.csv'
filename = 'EXACT_FE.csv'

dataframe = pd.DataFrame({'T':T_list, 'exact':FE_list})
#dataframe.columns = [ '%.3f'%a1  for a1 in T_list  ]    
dataframe.to_csv( filename )







print (XX)
print (YY)


plt.plot(XX,YY,'s-',label='exact',linewidth=2, markersize= 5)

plt.ylabel('$-\\beta f$',fontsize=14)
plt.xlabel('$T$',fontsize=14)

plt.legend(loc=0,fontsize=10)
plt.show()

