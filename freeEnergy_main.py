# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:33:36 2020

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
from pathlib import Path





def SAVE_dATA( Cdata,RG_step,a_list,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['L={}'.format(2**(num_steps+1)) for num_steps in range(RG_step) ]
    dataframe.columns = [ '%.3f'%a1  for a1 in a_list  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )




#================================================#
trgstep = 2
N = 2
step= 50
deltT = (4.0/step)
dcut = 


T_list  = [  1+deltT*i  for i in range(1,step+1) ]


FE_D = dict()


for T in T_list:

    DT, IDT = HOTRG_two.Ising_exact(2,1/T,'float')

 #   DT, IDT = HOTRG_two.Ising_square (T)

    T0 = DT;  iT0 = IDT; N_list = []; ti=0; FE_list = np.zeros(trgstep)
    for ii in range(trgstep):
        ## update along y-direction
        T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
        iT1 = HOTRG_two.updat_impurity( iT0, T0,'y',dcut,UU,UUT,N1)
        N_list.append(N1)

        ## update along x-direction
        T2,UU,UUT,N1 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
        iT2 = HOTRG_two.updat_impurity( iT1, T1,'x',dcut,UU,UUT,N1)
        N_list.append(N1)

        T0 = T2;  iT0 = iT2


        dab = T0.shape[0]*T0.shape[1]
        OSIT = np.trace(np.reshape(T0,(dab,dab)))

        va1 = 0.0; ss=0
        for nn in range( 2*(ii+1) ):
            va1 = va1+ np.log(N_list[nn])*(0.5**(nn+1))

        FE = -T*va1- T/(4**(ii+1))*np.log(OSIT)
        FE_list[ti] = FE;   ti+=1

    FE_D[T] = FE_list



#directory = './results_L%s_T%s' % (L,'%.3f'%T)
directory = './IsingFreeEnergy/RG{}'.format(trgstep)

dir_path = Path(directory)
if not dir_path.exists():
    dir_path.mkdir()

filename = directory +'/FE_D' + str(dcut) + '.csv'
SAVE_dATA( FE_D, trgstep,T_list,dcut,filename)










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
