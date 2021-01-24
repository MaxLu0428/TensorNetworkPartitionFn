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
from functools import lru_cache



def SAVE_dATA( Cdata,RG_step,a_list,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['L={}'.format(2**(num_steps+1)) for num_steps in range(RG_step) ]
    dataframe.columns = [ '%.6f'%a1  for a1 in a_list  ]

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
    T2 = np.transpose(T2,(0,2,1,3))
    dc = T2.shape[0]
    TM = np.reshape(T2,(dc*dc,dc*dc))
    return TM

#================================================#
trgstep = 15
N = 2
step= 40
deltT = (4.0/step)
dcut = 16
# dcut = #bond_dim#
FILE = './transfer_matrix_SingleColumn'
Tc = 2.0/np.log(1.0 + np.sqrt(2))



#FD_D = dict()




# T_list  = np.linspace(2.269195,2.269504,310)
Tc = 2/(np.log(1+2**(1/2)))
T_list = [Tc]
correlation_length = dict()


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

@lru_cache(None)
def RG(L):
    if L == 1:
        return T0,1
    Tleft ,  Nleft  = RG(L//2)
    Tright,  Nright = RG(L-L//2)
    Tmerge, _,_, Nmerge = HOTRG_two.updat_pure(Tleft,Tright,'y',dcut)
    return Tmerge,Nmerge
Lnum = 10
Ls = [i for i in range(1,Lnum+1)]
for T in T_list:
    CL_list = np.zeros(Lnum)
    DT, IDT = HOTRG_two.Ising_square(T,bias=0)
    T0 = DT;  iT0 = IDT; N_list = []; ti=0; FE_list = np.zeros(trgstep+1)
    for L in Ls:
        print(L)
        Ti,Ni = RG(L)
        TM = transfer_matrix( Ti )

        Y, Z = linalg.eigh( TM)
        # Y = np.abs(Y)
        lambda1,lambda2 = exclude_degenerate(Y)
        CL_list[L] = 1/np.log(np.abs(lambda1/lambda2))
    correlation_length[T] = CL_list;
SAVE_dATA(correlation_length, Ls,T_list,dcut,FILE+'/HOTRG_SingleColumn_'+str(dcut)+'.csv' )
