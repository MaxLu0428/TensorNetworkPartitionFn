# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:38:42 2020

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



def SAVE_dATA( Cdata,RG_step,Dcut,name):

    dataframe = pd.DataFrame( Cdata )
    dataframe.index=['i={}'.format(num_steps) for num_steps in range((2**RG_step)*2) ]
    dataframe.columns = [ a1  for a1 in  range( (2**RG_step)*2)  ]

    dataframe = dataframe.stack()
    dataframe = dataframe.unstack(0)
    dataframe.index.name="Dcut="+str(Dcut)
    dataframe.to_csv(name )



#================================================#
trgstep = 2
N = 2
dcut =  16


FILE = 'Ising_square_'+ 'RG'+str(trgstep)
step= 50
deltT = (2.0/step)
T_list  = [  1.0 +deltT*i  for i in range(0,step+1) ]



dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()


.
for T in T_list:
    Cij_list = dict();

    for i in range( (2**trgstep)*2):
        Cj_list = np.zeros(2**trgstep*2)

 #       for j in range((2**trgstep)*2):
        for j in range(i,(2**trgstep)*2):

            DT, IDT = HOTRG_two.Ising_square(T)
            To = DT;  Ti = IDT;

            si = i ; sj = j; Pi = Ti ; To0 = To

            if i==0 and j==0:
                Ti = To
                Pi = To

            for ii in range(trgstep):

                Lsi,si = divmod(si,2)
                Lsj,sj = divmod(sj,2)

    #
                ## update along y-direction
                To1,UU,UUT,N1 =  HOTRG_two.updat_pure2( To0,To0,'y',dcut)
                To1 /= N1
                if Lsi==0 and  Lsj==0:
                    T0,T1,T2,T3 =  HOTRG_two.DetT (si,sj,To0,Ti,Pi)
                else:
                    T0,T1,T2,T3 =  HOTRG_two.DetT (si,sj,To0,Ti,To0)

                iTL,_,_,_ =  HOTRG_two.updat_pure2( T0, T3,'y',dcut)
                iTL = iTL / N1
                iTR,_,_,_ =  HOTRG_two.updat_pure2( T1, T2,'y',dcut)
                iTR = iTR / N1


                if Lsi==0 and  Lsj==0:
                    P0 =To0; P1 = To0; P2 = To0; P3= To0
                else:
                    P0,P1,P2,P3 =  HOTRG_two.DetP (si,sj,To0,Pi)

                iPL,_,_,_ =  HOTRG_two.updat_pure2( P0, P3,'y',dcut)
                iPL = iPL / N1
                iPR,_,_,_ =  HOTRG_two.updat_pure2( P1, P2,'y',dcut)
                iPR = iPR / N1
                si = Lsi;  sj = Lsj;


                ## update along x-direction
                To2,UU,UUT,N1 =  HOTRG_two.updat_pure2( To1,To1,'x',dcut)
                To2 /= N1
                Tij,_,_,_ =  HOTRG_two.updat_pure2( iTL, iTR,'x',dcut)
                Tij = Tij / N1
                Pij,_,_,_ =  HOTRG_two.updat_pure2( iPL, iPR,'x',dcut)
                Pij = Pij / N1
                To0  =To2
                Ti = Tij
                Pi = Pij



            T0,T1,T2,T3 =  HOTRG_two.DetTLIST (si,sj,To0,Ti,Pi)

            if i==0 and j==0:
                T0 =Ti; T1=To0; T2=To0; T3=To0


            Norm =  HOTRG_two.merge_four(  [To0,To0,To0,To0] )
            Cij  =  HOTRG_two.merge_four(  [T0, T1, T2, T3])/Norm

            Cj_list[j] = Cij


        Cij_list[i] = Cj_list;




    SAVE_dATA( Cij_list, trgstep,dcut,FILE+'/Cij_T'+str(format( T, '.3f'))+'_D'+str(dcut)+'.csv' )
