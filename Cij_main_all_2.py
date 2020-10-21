# -*- coding: utf-8 -*-

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
trgstep = 1
N = 2
dcut =  16
T = 2./np.log(1+2**0.5)


FILE = './RG'+str(trgstep)
step= 50
deltT = (2.0/step)
T_list  = [  1.0 +deltT*i  for i in range(step+1) ]



dir_path = Path(FILE)
if not dir_path.exists():
    dir_path.mkdir()



for T in T_list:
    CorMatrix = np.zeros((2**trgstep*2,2**trgstep*2));
    DT, IDT = HOTRG_two.Ising_exact(2,1/T,'float')
    To = DT;  Ti = IDT;
    # try to initiate memo
    memo = HOTRG_two.creatMemo(trgstep,sites=[(0,1)])
    memo['denominator'].append(To)
    if (0,0) in memo:
        memo[(0,0)].append(To)
    for ele in memo:
        if ele != 'denominator' and ele != (0,0):
            memo[ele].append(Ti)
    for ii in range(trgstep+1):
        HasUpdated = set()
        To1,UU,UUT,N1 =  HOTRG_two.updat_pure(memo['denominator'][-1],
                                        memo['denominator'][-1],'y',dcut)
        To2,UU2,UUT2,N2 =  HOTRG_two.updat_pure( To1,To1,'x',dcut)
        for graph in [5,6,7,1,2,3,4]:
            for site in memo :
                if memo[site][ii] == graph and site not in HasUpdated:
                    T0,T1,T2,T3 = HOTRG_two.graph_tensor_map(graph,memo['denominator'][-1]
                    ,memo['one side'][-1],memo[site][-1])
                    if ii == trgstep: # final step of HOTRG -- take tensor trace
                        Norm = HOTRG_two.merge_four([memo['denominator'][-1],memo['denominator'][-1]
                                            ,memo['denominator'][-1],memo['denominator'][-1]])
                        Cij  = HOTRG_two.merge_four([T0,T1,T2,T3]) / Norm
                        if site != 'one side':
                            CorMatrix[site] = Cij
                    else:
                        iPL =  HOTRG_two.updat_impurity( T0, T3,'y',dcut,UU,UUT,N1)
                        iPR =  HOTRG_two.updat_impurity( T1, T2,'y',dcut,UU,UUT,N1)
                        ## update along x-direction
                        Pij =  HOTRG_two.updat_impurity( iPL, iPR,'x',dcut,UU2,UUT2,N2)
                        for s2 in memo:
                            if s2 != site and memo[s2][-1] is memo[site][-1]:
                                if memo[s2][ii] == memo[site][ii]:
                                    memo[s2][-1] = Pij
                                    HasUpdated = HasUpdated | {s2}
                        memo[site][-1] = Pij
                        HasUpdated = HasUpdated | {site}
        memo['denominator'][-1] = To2

    SAVE_dATA( CorMatrix, trgstep,dcut,FILE+'/Cij2_T'+str(format( T, '.3f'))+'_D'+str(dcut)+'.csv' )
