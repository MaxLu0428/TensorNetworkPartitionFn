#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12

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
    dataframe.index=['RG steps={}'.format(num_steps) for num_steps in range(RG_step) ]
    dataframe.columns = [ '%.3f'%a1  for a1 in a_list  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )



#================================================#
dcut = 8
trgstep = 5
N = 2
step= 50
deltT = (2.0/step)
FILE = './DATA_2'

dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()



T_list  = [ 1.0+ deltT*i  for i in range(1,step+1) ]


for dcut in [4,8,12,16,20,24,32,40]:

    M2_D = dict(); M4_D = dict();

    for T in T_list:
        M2_list = np.zeros(trgstep); M4_list = np.zeros(trgstep)

        DT, IDT = HOTRG_two.Ising_exact(2,1/T,'float')

        S0 = DT;  S1 = IDT ; S2 = DT;  S3 = IDT ; S4 = DT;
        for ii in range(trgstep):

            S0,S1,S2,S3,S4 = HOTRG_two.M4_calculation (S0,S1,S2,S3,S4,dcut)

            dab = S0.shape[0]*S0.shape[1]
            norm = np.trace(np.reshape(S0,(dab,dab)))
            M2 =  np.trace(np.reshape( S2,(dab,dab)))/norm
            M4 =  np.trace(np.reshape( S4,(dab,dab)))/norm

            M2_list[ii] = M2; M4_list[ii] = M4

        M2_D[T] = M2_list; M4_D[T] = M4_list


    SAVE_dATA( M2_D, trgstep,T_list,dcut,FILE+'/M2_D'+str(dcut)+'.csv' )
    SAVE_dATA( M4_D, trgstep,T_list,dcut,FILE+'/M4_D'+str(dcut)+'.csv' )
