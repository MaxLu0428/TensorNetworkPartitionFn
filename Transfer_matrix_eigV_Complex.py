#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    dataframe.columns = [ a1  for a1 in range(1,a_list+1) ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name,index=False)


def SAVE_dATA2( Cdata,index,Lnum,Dcut,name):
    columns = []
    for num in range(1,Lnum+1):
        columns.append(num)
    dataframe = pd.DataFrame( Cdata )
    dataframe.index= [index]
    dataframe.columns = columns
    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name,index=False)

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
dcut=512
# dcut = #bond_dim#
FILE = 'transfer_matrix_SingleColumn_ComplexH_Eig'
FILE2 = 'transfer_matrix_SingleColumn_ComplexH_NormalizationFactor'

makeFolder(FILE)
makeFolder(FILE2)

Tc = 2.0/np.log(1.0 + np.sqrt(2))

Tc = 2/(np.log(1+2**(1/2)))
Kc = np.log(1+np.sqrt(2))/2
EigVnums = dcut
T_list = [Tc*3]
# h_list = np.linspace(0.39,0.399,10)
h_list=[0.392,0.393]
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

Lnum = 11
Ls = [i for i in range(1,Lnum+1)]
scaleFactors = np.zeros((len(T_list),Lnum))
for h in h_list:
    Normalization_map = {1:0}
    CL_list = np.zeros(Lnum)
    eigenvalues = np.zeros((Lnum,EigVnums))
    DT, IDT = HOTRG_two.Ising_square(Tc*3,bias=0,h=h)
    T0 = DT;  iT0 = IDT; N_list = [];
    for i,L in enumerate(Ls):


        Ti,Ni = RG(L)

        if L not in Normalization_map:
            Normalization_map[L] = [np.log(Ni)]
        TM = transfer_matrix( Ti )

        Y, Z = linalg.eigh( TM)

        if len(Y) < EigVnums:

            totalEigs = len(Y)
            Num0 = EigVnums - totalEigs
            remains0 = [ 0 for i in range(Num0)]
            Ys = list(Y)[::-1]
            Ys = np.log(np.array(Ys))
            Ys = list(Ys)
            eigenvalues[i,:] = np.array(Ys + remains0)
        else:
            Ys = Y[:-1*EigVnums-1:-1]
            Ys = np.log(Ys)
            eigenvalues[i,:] = Ys
    RG.cache_clear()
    print(eigenvalues[:4,:2])
    # SAVE_dATA(eigenvalues, Lnum,EigVnums,dcut,'./'+FILE+'/h{}_Eigenvalue_Dcut{}.csv'.format(h,dcut))
    # SAVE_dATA2(Normalization_map,h,Lnum,dcut,'./'+FILE+'/h{}_NormalizationFactor_Dcut{}.csv'.format(h,dcut))
