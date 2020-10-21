 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:28:06 2020

@author: chingyuhuang
"""



import numpy as np
from math import pi
from scipy import linalg
import time, itertools
import matplotlib.pyplot as plt
import HOTRG_two
import pandas as pd
import os




  
    
def SAVE_dATA( Cdata,RG_step,a_list,Dcut,name):
    
    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['L={}'.format(2**(num_steps+1)) for num_steps in range(RG_step) ]
    dataframe.columns = [ '%.3f'%a1  for a1 in a_list  ] 
    
    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )

def transfer_matrix(T):
    T2 = np.tensordot(T,T,axes=([1,3],[3,1]) )
    T4 = np.tensordot(T2,T2,axes=([0,2],[1,3]) )
    T4 = np.transpose( T4, (0,2,1,3) )
    dc = T4.shape[0]
    TM = np.reshape(T4,(dc*dc, dc*dc) )
    return TM


#================================================#
trgstep = 2
N = 2
step= 40
deltT = (4.0/step)
dcut = 16

Tc = 2.0/np.log(1.0 + np.sqrt(2)) 



#FD_D = dict()
DT, IDT = HOTRG_two.Ising_exact(2,1/Tc,'float')

T0 = DT;  iT0 = IDT; N_list = []; ti=0; FE_list = np.zeros(trgstep)
N1 = np.max(abs(T0))
T0 = T0/N1
N_list.append(N1)

TM = transfer_matrix( T0 )
Y, Z = linalg.eigh( TM)
print (Y)


for RG in range(trgstep):
    
    T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
    iT1 = HOTRG_two.updat_impurity( iT0, T0,'y',dcut,UU,UUT,N1)
    N_list.append(N1)
            
        ## update along x-direction
    T2,UU,UUT,N1 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
    iT2 = HOTRG_two.updat_impurity( iT1, T1,'x',dcut,UU,UUT,N1)     
    N_list.append(N1)
        
    T0 = T2;  iT0 = iT2
    
    TM = transfer_matrix( T0 )
    Y, Z = linalg.eigh( TM)
    
    print (Y)
    
    
    
#directory = './results_L%s_T%s' % (L,'%.3f'%T)
#directory = './Ising'  
#os.makedirs(directory)
#filename = directory +'/FE_D' + str(dcut) + '.csv'
#SAVE_dATA( FE_D, trgstep,T_list,dcut,filename)
    
        
    
    
    
    
    
    


#
#    
#    
#clist=['r','k','g','b','y','lightsalmon','silver','lime','steelblue','olive']
#makerlist =['o','x','+','D','s','^','<','>','o','x']
#    
#f = plt.figure(figsize=(8, 6)); # plot the calculated values    
#ax = plt.axes([.15, .12, .7, .78])
#
#plt.plot(T_list, MZlist,c = clist [0], ls='--',lw= 1.0,    ##label='$\mathcal{O} (n!)$',
#         marker = makerlist[0], mec= clist[0], mfc= 'none', mew = 1.2,  ms = 7)  
#
#
#
#plt.xlabel("$n$", fontsize=20);
##plt.ylabel(" ", fontsize=20);       
#plt.axis('tight')
#plt.legend( loc=1, fontsize=14)
##plt.xlim(0,val)
##plt.ylim(0,5000)
##plt.title('$HOTRG \chi$='+str(chi_max), fontsize= 20)  
##plt.text(0.1,0.1, 'Lowest : -0.6167', transform=ax.transAxes, ha='left', fontsize=15)
#   
#plt.show()
#
#
#
