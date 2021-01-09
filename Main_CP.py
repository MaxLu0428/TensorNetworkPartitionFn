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
      


#================================================#
trgstep = 10
N = 2
dcut = 16


filename1 = 'TM_EVEN_D'+str(dcut)+'.txt'
with open(filename1, 'w') as f:
#    f.write('# correlation_length v.s m \n')
    f.write('# RG \t L \t Real_L \t norm \t E-0 \t E1 \t E2 \t E3 \n')
f.close()


filename2 = 'TM_ODD_D'+str(dcut)+'.txt'
with open(filename2, 'w') as f:
#    f.write('# correlation_length v.s m \n')
    f.write('# RG \t L \t Real_L \t norm \t E-0 \t E1 \t E2 \t E3 \n')
f.close()







Tc = 2.0/np.log(1.0 + np.sqrt(2)) 
DT, IDT = HOTRG_line.Ising_exact(2,1/Tc,'float')


T0 = DT; N_list = [];
N1 = np.max(abs(T0))
T0 = T0/N1
N_list.append(N1)


qun = [0,1]

Mat = HOTRG_line.transfer_matrix( T0, qun )






f = open(filename1, 'a')
message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
           .format( 0,1, 1 , N1 , Mat[0][0], 0. ,0., 0.)
f.write(message); f.write('\n');f.close()




f = open(filename2, 'a')
message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
           .format( 0,1, 1 , N1 , Mat[1][0], 0. ,0., 0.)
f.write(message); f.write('\n');f.close()





Tt = DT;
N1 = np.max(abs(Tt))
Tt = Tt/N1
a_label = [0,1]

b_label = [0,1]



for RG in range(trgstep):
    
    
    #print ('-------------',RG+1)
    
    #### x-direction
    
    
    dimTt = Tt.shape
    Aup = np.tensordot( Tt, Tt.conjugate(), axes= ([2,3],[2,3]))
    Adown= np.tensordot( Tt, Tt.conjugate(), axes= ([2,1],[2,1]))
    AA =  np.tensordot( Aup, Adown, axes= ([1,3],[1,3]))
    AA =  np.reshape( np.transpose (AA,(0,2,1,3)),(dimTt[0]*dimTt[0],dimTt[0]*dimTt[0]))
    
    
    theta = AA
    UU, a_label = HOTRG_line.determine_U ( theta, a_label,dcut )
    Tt = HOTRG_line.merge_two (Tt, Tt, UU, UU.T.conj() ) 
    
    #### y-direction ###############
    
    
    Tt = np.transpose( Tt,(1,0,3,2) )
    
    dimTt = Tt.shape
    Aup = np.tensordot( Tt, Tt.conjugate(), axes= ([2,3],[2,3]))
    Adown= np.tensordot( Tt, Tt.conjugate(), axes= ([2,1],[2,1]))
    AA =  np.tensordot( Aup, Adown, axes= ([1,3],[1,3]))
    AA =  np.reshape( np.transpose (AA,(0,2,1,3)),(dimTt[0]*dimTt[0],dimTt[0]*dimTt[0]))
    
    theta = AA
    UU, b_label = HOTRG_line.determine_U ( theta, b_label,dcut )
    Tt = HOTRG_line.merge_two (Tt, Tt, UU, UU.T.conj() ) 
    Tt  = np.transpose(Tt,(1,0,3,2))
    
    
    

    
    
    #######   transfer matrix  #######
    
    N1 = np.max(abs(Tt))
    Tt = Tt/N1
    N_list.append(N1)
    

    
    

    
    
    
    Mat = HOTRG_line.transfer_matrix( Tt,a_label )



    
    EVV = np.zeros(4 )
    ODD = np.zeros(4 )
    
    for i in range( np.min( [4, len(Mat[0]) ] )   ):
        EVV[i] = Mat[0][i]
        
    for i in range( np.min( [4, len(Mat[1]) ] )   ):
        ODD[i] = Mat[1][i]
        

    
#    print (RG,N1, Mat)
    
    
    f = open(filename1, 'a')
    message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
    .format( RG+1,1 , 2**(RG+1), N1 , EVV[0], EVV[1] , EVV[2], EVV[3])
    f.write(message); f.write('\n');f.close()
        
        
        
        
    f = open(filename2, 'a')
    message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
    .format( RG+1,1 , 2**(RG+1), N1 , ODD[0], ODD[1] , ODD[2] ,ODD[3])
    f.write(message); f.write('\n');f.close()  
    
    
    
    
#    if RG==0: 
#        
#        f = open(filename1, 'a')
#        message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
#                   .format( RG+1,1 , 2**(RG+1), N1 , Mat[0][0], Mat[0][1] ,0., 0.)
#        f.write(message); f.write('\n');f.close()
#        
#        
#        
#        
#        f = open(filename2, 'a')
#        message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
#                   .format( RG+1,1 , 2**(RG+1), N1 , Mat[1][0], Mat[1][1] ,0., 0.)
#        f.write(message); f.write('\n');f.close()
#        
#    else:
#            
#        f = open(filename1, 'a')
#        message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
#                   .format(RG+1,1 , 2**(RG+1), N1 , Mat[0][0], Mat[0][1] ,Mat[0][2], Mat[0][3])
#        f.write(message); f.write('\n');f.close()
#        
#        
#        
#        
#        f = open(filename2, 'a')
#        message = ('{:2d} ' +'{:1d} ' +'{:10d} ' + '{:.10f} ' + '{:.10f} '+  '{:.10f} '+'{:.10f} '+'{:.10f} ')\
#                   .format(RG+1,1 , 2**(RG+1), N1 , Mat[1][0], Mat[1][1] ,Mat[1][2], Mat[1][3])
#        f.write(message); f.write('\n');f.close()
#         
        
   


