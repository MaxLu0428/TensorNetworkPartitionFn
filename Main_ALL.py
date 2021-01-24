 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:28:06 2020

@author: chingyuhuang
"""



import numpy as np
from math import pi
from scipy import linalg
import HOTRG_line
import scipy.sparse as sparse
import scipy
from scipy.sparse.linalg import eigs
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




FILE = './DATA1'

dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()

#================================================#
trgstep = 20
N = 2
dcut = 48



Tc = 2.0/np.log(1.0 + np.sqrt(2))
step= 50
iniT = 2.0
deltT = (0.5/step)
T_list  = [ iniT + deltT*i  for i in range(1,step+1) ]





E1_ev = dict(); E2_ev = dict();
E3_ev = dict(); E4_ev = dict();

E1_odd = dict(); E2_odd = dict();
E3_odd = dict(); E4_odd = dict();

Norm  = dict()


for T in T_list:
    Norm_list = np.zeros(trgstep+1);
    E1_list = np.zeros( (2,trgstep+1));
    E2_list = np.zeros((2,trgstep+1));
    E3_list = np.zeros((2,trgstep+1));
    E4_list = np.zeros((2,trgstep+1));

    DT, IDT = HOTRG_line.Ising_exact(2,1/T,'float')


    T0 = DT; N_list = [];
    N1 = np.max(abs(T0))
    T0 = T0/N1
    N_list.append(N1)

    qun = [0,1]
    Mat = HOTRG_line.transfer_matrix( T0, qun )


    EVV = np.zeros(4 )
    ODD = np.zeros(4 )

    for i in range( np.min( [4, len(Mat[0]) ] )   ):
        EVV[i] = Mat[0][i]
    for i in range( np.min( [4, len(Mat[1]) ] )   ):
        ODD[i] = Mat[1][i]



    Norm_list[ 0 ] = N1
    E1_list [0,0] = EVV[0]
    E2_list [0,0] = EVV[1]
    E3_list [0,0] = EVV[2]
    E4_list [0,0] = EVV[3]

    E1_list [1,0] = ODD[0]
    E2_list [1,0] = ODD[1]
    E3_list [1,0] = ODD[2]
    E4_list [1,0] = ODD[3]





    Tt = DT;
    N1 = np.max(abs(Tt))
    Tt = Tt/N1
    a_label = [0,1]

    for RG in range(trgstep):


        print ('-------------',RG+1)


        dimTt = Tt.shape
        Aup = np.tensordot( Tt, Tt.conjugate(), axes= ([2,3],[2,3]))
        Adown= np.tensordot( Tt, Tt.conjugate(), axes= ([2,1],[2,1]))
        AA =  np.tensordot( Aup, Adown, axes= ([1,3],[1,3]))
        AA =  np.reshape( np.transpose (AA,(0,2,1,3)),(dimTt[0]*dimTt[0],dimTt[0]*dimTt[0]))


        theta = AA
            # start SVD
        UU, a_label = HOTRG_line.determine_U ( theta, a_label,dcut )



        Tt = HOTRG_line.merge_two (Tt, Tt, UU, UU.T.conj() )


        N1 = np.max(abs(Tt))
        Tt = Tt/N1
        N_list.append(N1)


        Mat = HOTRG_line.transfer_matrix( Tt, a_label )


        EVV = np.zeros(4 )
        ODD = np.zeros(4 )

        for i in range( np.min( [4, len(Mat[0]) ] )   ):
            EVV[i] = Mat[0][i]

        for i in range( np.min( [4, len(Mat[1]) ] )   ):
            ODD[i] = Mat[1][i]


        E1_list [0,RG+1] = EVV[0]
        E2_list [0,RG+1] = EVV[1]
        E3_list [0,RG+1] = EVV[2]
        E4_list [0,RG+1] = EVV[3]

        E1_list [1,RG+1] = ODD[0]
        E2_list [1,RG+1] = ODD[1]
        E3_list [1,RG+1] = ODD[2]
        E4_list [1,RG+1] = ODD[3]
        Norm_list[ RG+1 ] = N1

    E1_ev[ T] = E1_list[0]
    E2_ev[ T] = E2_list[0]
    E3_ev[ T] = E3_list[0]
    E4_ev[ T] = E4_list[0]
    Norm[ T] = Norm_list


    E1_odd[ T] = E1_list[1]
    E2_odd[ T] = E2_list[1]
    E3_odd[ T] = E3_list[1]
    E4_odd[ T] = E4_list[1]


SAVE_dATA( E1_ev, trgstep,T_list,dcut,FILE+'/E1ev_D'+str(dcut)+'.csv' )
SAVE_dATA( E2_ev, trgstep,T_list,dcut,FILE+'/E2ev_D'+str(dcut)+'.csv' )
SAVE_dATA( E3_ev, trgstep,T_list,dcut,FILE+'/E3ev_D'+str(dcut)+'.csv' )
SAVE_dATA( E4_ev, trgstep,T_list,dcut,FILE+'/E4ev_D'+str(dcut)+'.csv' )
SAVE_dATA( Norm, trgstep,T_list,dcut,FILE+'/Norm_X'+str(dcut)+'.csv' )


SAVE_dATA( E1_odd, trgstep,T_list,dcut,FILE+'/E1odd_D'+str(dcut)+'.csv' )
SAVE_dATA( E2_odd, trgstep,T_list,dcut,FILE+'/E2odd_D'+str(dcut)+'.csv' )
SAVE_dATA( E3_odd, trgstep,T_list,dcut,FILE+'/E3odd_D'+str(dcut)+'.csv' )
SAVE_dATA( E4_odd, trgstep,T_list,dcut,FILE+'/E4odd_D'+str(dcut)+'.csv' )
