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
from pathlib import Path

def SAVE_dATA( Cdata,RG_step,a_list,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['L={}'.format(2**(num_steps+1)) for num_steps in range(RG_step) ]
    dataframe.columns = [ '%.6f'%a1  for a1 in a_list  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )


def transfer_matrix(T):
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
dcut = #bond_dim#
FILE = './transfer_matrix_M2'
Tc = 2.0/np.log(1.0 + np.sqrt(2))



#FD_D = dict()




T_list  = np.linspace(1,3,100)

M2s = dict()


def exclude_degenerate(Y,tolerence = 0):
    largest = Y[-1]
    lambda1s = [largest]
    for eg in Y[-2::-1]:
        if np.abs(eg-largest) < tolerence:
            lambda1s.append(eg)
        else:
            lambda2 = eg
            lambda1 = sum(lambda1s) / len(lambda1s)
            return lambda1,lambda2

def getM2(T0,iT0):
    T0_1d  = transfer_matrix(T0)
    iT0_1d = transfer_matrix(iT0)
    _, U = linalg.eigh(T0_1d)
    denominator = np.matmul(U.T,np.matmul(T0_1d,U))
    numerator   = np.matmul(U.T,np.matmul(iT0_1d,U))
    return np.trace(np.matmul(numerator,numerator)) / np.trace(np.matmul(denominator,denominator))


def makeFolder(Filename,sep='/'):
   '''file name may contains several folders, so need to check
   individually'''
   subFolderPath = Filename.split('/')
   complete = '.'
   for subFolderP in subFolderPath:
       complete = "/".join([complete,subFolderP])
       dir_path = Path(complete)
       if not dir_path.exists():
           dir_path.mkdir()

makeFolder(FILE)

for T in T_list:
    M2_list = np.zeros(trgstep+1)
    DT, IDT = HOTRG_two.Ising_square(T,bias=0)
    T0 = DT;  iT0 = IDT; N_list = []; ti=0; FE_list = np.zeros(trgstep+1)
    for RG in range(1,trgstep+1):
        T1,UU,UUT,N1 = HOTRG_two.updat_pure( T0,T0,'y',dcut)
        iT1 = HOTRG_two.updat_impurity( iT0, T0,'y',dcut,UU,UUT,N1)
        ## update along x-direction
        T2,UU,UUT,N1 = HOTRG_two.updat_pure( T1,T1,'x',dcut)
        iT2 = HOTRG_two.updat_impurity( iT1, T1,'x',dcut,UU,UUT,N1)
        T0 = T2;  iT0 = iT2

        M2_list[RG] = getM2(T0,iT0)
    M2s[T] = M2_list
SAVE_dATA(M2s, trgstep+1,T_list,dcut,FILE+'/M2_'+str(dcut)+'.csv' )
