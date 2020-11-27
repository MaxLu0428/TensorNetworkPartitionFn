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
    dataframe.columns = [ '%.4f'%a1  for a1 in a_list  ]

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

def transfer_matrix2(T):
    ''' instead of 2x2 unitcell , try to use 2x1 '''
    T2 = np.tensordot(T,T,axes=([1,3],[3,1]))
    dc = T2.shape[0]
    TM = np.reshape(T2,(dc*dc,dc*dc))
    return TM

def transfer_matrix3(T):
    dims = T.shape
    result = np.zeros((dims[0],dims[2]))

    for i in range(dims[0]):
        for j in range(dims[2]):
            result[i][j] = np.trace(T[i,:,j,:])
    return result

#================================================#
trgstep = 15
N = 2
step= 40
deltT = (4.0/step)
dcut = 16
FILE = './transfer_matrix'
Tc = 2.0/np.log(1.0 + np.sqrt(2))



#FD_D = dict()




T_list  = [Tc]
correlation_length = dict()

for T in T_list:
    CL_list = np.zeros(trgstep)
    DT, IDT = HOTRG_two.Ising_square(T,bias=10**-5)
    T0 = DT;  iT0 = IDT; N_list = []; ti=0; FE_list = np.zeros(trgstep)
    N1 = np.max(abs(T0))
    T0 = T0/N1
    N_list.append(N1)
    TM = transfer_matrix( T0 )
    Y, Z = linalg.eigh( TM)
    for RG in range(trgstep):

        T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
        iT1 = HOTRG_two.updat_impurity( iT0, T0,'y',dcut,UU,UUT,N1)
        N_list.append(N1)
        ## update along x-direction
        T2,UU,UUT,N1 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
        iT2 = HOTRG_two.updat_impurity( iT1, T1,'x',dcut,UU,UUT,N1)
        N_list.append(N1)
        T0 = T2;  iT0 = iT2
        TM = transfer_matrix3( T0 )
        Y, Z = linalg.eigh( TM)
        print("L={}".format(2**RG*2))

        lambda1 = Y[-1]
        lambda2 = Y[-2]
        print('correlation length ')
        print(1/np.log((lambda1/lambda2)))
        CL_list[RG] = 1/np.log(np.abs(lambda1/lambda2))
    correlation_length[T] = CL_list;
# SAVE_dATA(correlation_length, trgstep,T_list,dcut,FILE+'/_D'+str(dcut)+'.csv' )
