import matplotlib.pyplot as plt
import time
import numpy as np
from math import pi
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import csv


RG = 1






dcut = 16


step = 50;  
deltT = (0.5/step)
T_list  = [ 2.0 + deltT*i  for i in range(1,step+1) ]

fig = plt.figure()
clist=['r','k','g','b','y','lightsalmon','silver','lime','steelblue','olive','r']
makerlist =['o','x','+','D','s','^','<','>','o','x','+']



ii = 0
for RG in [1,2,3,4,5,6]:

    Size = 2**(RG+1)
    file = 'RG'+str(RG)+'/'
    Cij_list = np.zeros( ( Size+1 , Size))
    q1 = 2*pi/Size
    Xia_list =[]
    Xib_list =[]    
    
    for T in T_list:
        
        NAME = file+'Cij_T'+str(  '%.3f'%T   )+'_D'+str(dcut)+'.csv'
        
    
        with open( NAME, newline='') as csvfile:
            
            rows = csv.reader(csvfile)
            a = 0
            for row in rows:
                if a>0: Cij_list[:,a-1] = row
                a+=1
                
        SO = 0
        SQ1 = 0              
        SQ2 = 0   
        for x in range(Size):
            for y in range(x,Size):
                Cxy = Cij_list[y+1,x]
              
                if x==y: 
                    SO += Cxy
                    SQ1 += np.cos(q1*x)*Cxy                
                    SQ2 += np.cos(2*q1*x)*Cxy 
                else: 
                    SO += 2*Cxy
                    SQ1 += np.cos(q1*x)*Cxy+ np.cos(q1*y)*Cxy             
                    SQ2 += np.cos(2*q1*x)*Cxy+ np.cos(2*q1*y)*Cxy  
                    
        Xia_list.append(  np.sqrt( SO/SQ1 -1 )/q1/Size )
        Xib_list.append(  np.sqrt( (SQ1/SQ2 -1)/(4-SQ1/SQ2) )/q1/Size )            


    plt.plot(  T_list, Xia_list,label= 'L='+str(Size),
             c = clist[ii],ls= '--', lw= 0.5, marker = makerlist[ii],
              mec= clist[ii], mfc= 'none', mew = 1.5, ms = 6)
    
    ii +=1
    
        
    plt.legend(loc=0,fontsize=12)
    plt.yscale("log")
    #plt.xscale("log")
    
    plt.title('$D_c=$'+str(dcut))
    #plt.ylim(0,1.0)
    
    plt.xlabel(' $T$', fontsize=20)
    plt.ylabel('$\\xi_a/L$' ,fontsize=20)


plt.show()







