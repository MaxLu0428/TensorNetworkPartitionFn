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
from pathlib import Path
import mpmath

def SAVE_dATA( Cdata,Lnum,a_list,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['L={}'.format(L) for L in range(1,Lnum+1) ]
    dataframe.columns = [ '%.6f'%a1  for a1 in a_list ]

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

#================================================#
trgstep = 15
N = 2
step= 40
deltT = (4.0/step)
dcut = 16
# dcut = #bond_dim#
FILE = './transfer_matrix_SingleColumn'
makeFolder(FILE)
Tc = 2.0/np.log(1.0 + np.sqrt(2))



#FD_D = dict()




# T_list  = np.linspace(2.269195,2.269504,310)
Tc = 2/(np.log(1+2**(1/2)))

T_list = [1]
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
        return DT,1
    Tleft ,  Nleft  = RG(L//2)
    Tright,  Nright = RG(L-L//2)
    Tmerge, _,_, Nmerge = HOTRG_two.updat_pure(Tleft,Tright,'y',dcut)
    return Tmerge,Nmerge

Lnum = 10
Ls = [i for i in range(1,Lnum+1)]
for T in T_list:
    CL_list = np.zeros(Lnum)
    DT, IDT = HOTRG_two.Ising_square(T,bias=0)
    T0 = DT;  iT0 = IDT; N_list = [];
    for i,L in enumerate(Ls):
        print(L)
        Ti,Ni = RG(L)
        TM = transfer_matrix( Ti )

        Y, Z = linalg.eigh( TM)
        # Y = np.abs(Y)
        lambda1,lambda2 = exclude_degenerate(Y)
        CL_list[i] = 1/np.log(np.abs(lambda1/lambda2))
    correlation_length[T] = CL_list;
SAVE_dATA(correlation_length, Lnum,T_list,dcut,FILE+'/HOTRG_SingleColumn_'+str(dcut)+'.csv' )
