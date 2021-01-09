 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:28:06 2020

@author: chingyuhuang
"""


from pathlib import Path
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
    dataframe.index=['L={}x{}'.format(2**(num_steps),2**(num_steps)) for num_steps in range(RG_step) ]
    dataframe.columns = [ a1  for a1 in range(1,a_list+1)  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )

def SAVE_dATA2( Cdata,Ts,RG_step,Dcut,name):
    columns = []
    for num_steps in range(RG_step):
        columns.append('L={}'.format(2**(num_steps)))
    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['T=Tc']
    dataframe.columns = columns
    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )

def transfer_matrix(T):
    T2 = np.tensordot(T,T,axes=([1,3],[3,1]) )
    T4 = np.tensordot(T2,T2,axes=([0,2],[1,3]) )
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
trgstep = 20
N = 2
step= 40
deltT = (4.0/step)
dcut = #bond_dim#
FILE = './transfer_matrix_eigenvalue_divL/'
FILE2 = './transfer_matrix_rescaleFactor_divL/'
Tc = 2.0/np.log(1.0 + np.sqrt(2))

dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()

dir_path2 = Path(FILE2)
if not dir_path.exists():
    dir_path2.mkdir()




# T_list  = np.linspace(2.0,2.5,51)
T_list = [Tc]
EigVnums = dcut
scaleFactors = np.zeros((len(T_list),1+trgstep))

for i,T in enumerate(T_list):
    eigenvalues = np.zeros((trgstep,EigVnums))
    DT, IDT = HOTRG_two.Ising_square(T,bias=0)
    T0 = DT;  iT0 = IDT; N_list = [1]; ti=0; FE_list = np.zeros(trgstep)
    for RG in range(trgstep):
        TM = transfer_matrix3( T0 )
        Y, Z = linalg.eigh(TM)
        if len(Y) < EigVnums:
            totalEigs = len(Y)
            Num0 = EigVnums - totalEigs
            remains0 = [ 0 for i in range(Num0)]
            Ys = list(Y)[::-1]
            Ys = np.log(np.array(Ys))/ (2**RG)
            Ys = list(Ys)
            eigenvalues[RG,:] = np.array(Ys + remains0)
        else:
            Ys = Y[:-1*EigVnums-1:-1]
            Ys = np.log(Ys)/(2**RG)
            eigenvalues[RG,:] = Ys
        T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
        T1 *= N1
        T2,UU,UUT,N2 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
        N_list.append(N2)
        T0 = T2


    scaleFactors[i,:] = N_list
    SAVE_dATA(eigenvalues,trgstep,EigVnums,dcut,FILE+'unit1x1_Tc'+'_D'+str(dcut)+'.csv' )
SAVE_dATA2(scaleFactors,T_list,1+trgstep,dcut,FILE2+'unit1x1_Tc_D'+str(dcut)+'.csv' )
