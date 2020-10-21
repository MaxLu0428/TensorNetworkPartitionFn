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
from pathlib import Path





def SAVE_dATA( Cdata,RG_step,a_list,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['RG steps={}'.format(num_steps+1) for num_steps in range(RG_step+1) ]
    dataframe.columns = [ '%.3f'%a1  for a1 in a_list  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )




#================================================#
trgstep = 10
N = 2
step= 40
deltT = (4.0/step)
FILE = './DATA_2'

dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()

T_list  = [  deltT*i  for i in range(1,step+1) ]


for dcut in [4,8,12,16,20,24,32,40]:

    CmD = dict(); ChD = dict(); RD  = dict()
    CmX = dict(); ChX = dict(); RX  = dict()

    for T in T_list:
        CmD_list = np.zeros(trgstep+1); ChD_list = np.zeros(trgstep+1)
        CmX_list = np.zeros(trgstep+1); ChX_list = np.zeros(trgstep+1)

        DT, IDT = HOTRG_two.Ising_exact(2,1/T,'float')

        Cm_D,Ch_D,Cm_X,Ch_X = HOTRG_two.Cmax_CHmax( DT,IDT,dcut)

        CmD_list[0] = Cm_D; ChD_list[0] = Ch_D
        CmX_list[0] = Cm_X; ChX_list[0] = Ch_X


        T0 = DT;  iT0 = IDT;
        for ii in range(trgstep):

            ## update along y-direction
            T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
            iT1 = HOTRG_two.updat_impurity( iT0, T0,'y',dcut,UU,UUT,N1)


            ## update along x-direction
            T2,UU,UUT,N1 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
            iT2 = HOTRG_two.updat_impurity( iT1, T1,'x',dcut,UU,UUT,N1)

            Cm_D,Ch_D,Cm_X,Ch_X = HOTRG_two.Cmax_CHmax( T2,iT2,dcut)
            CmD_list[ii+1] = Cm_D; ChD_list[ii+1] = Ch_D
            CmX_list[ii+1] = Cm_X; ChX_list[ii+1] = Ch_X

            T0  =T2
            iT0 =iT2

        CmD[T] = CmD_list; CmX[T] = CmX_list
        ChD[T] = ChD_list; ChX[T] = ChX_list


    SAVE_dATA( ChD, trgstep,T_list,dcut,FILE+'/ChM_D'+str(dcut)+'.csv' )
    SAVE_dATA( CmD, trgstep,T_list,dcut,FILE+'/CM_D'+str(dcut)+'.csv' )
    SAVE_dATA( ChX, trgstep,T_list,dcut,FILE+'/ChM_X'+str(dcut)+'.csv' )
    SAVE_dATA( CmX, trgstep,T_list,dcut,FILE+'/CM_X'+str(dcut)+'.csv' )









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
