#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:38:42 2020

@author: chingyuhuang
"""


import numpy as np
from math import pi
from scipy import linalg



#==================================================================#
def DetT (si,sj,To,Ti,Pi):

    if si ==0 and sj == 0:
        T0 =Ti; T1=To; T2=To; T3=To
    elif si ==1 and sj == 0:
        T0 =Ti; T1=Pi; T2=To; T3=To
    elif si ==0 and sj == 1:
        T0 =Ti; T1=To; T2=To; T3=Pi
    elif si ==1 and sj == 1:
        T0 =Ti; T1=To; T2=Pi; T3=To

    return T0,T1,T2,T3
#==================================================================#


#==================================================================#
def DetP (si,sj,To,Pi):

    if si ==0 and sj == 0:
        T0 =Pi; T1=To; T2=To; T3=To
    elif si ==1 and sj == 0:
        T0 =To; T1=Pi; T2=To; T3=To
    elif si ==0 and sj == 1:
        T0 =To; T1=To; T2=To; T3=Pi
    elif si ==1 and sj == 1:
        T0 =To; T1=To; T2=Pi; T3=To

    return T0,T1,T2,T3
#==================================================================#


#==================================================================#
def DetTLIST (si,sj,To,Ti,Pi):

    if si ==0 and sj == 0:
        T0 =Ti; T1=To; T2=To; T3=To
    elif si ==1 and sj == 0:
        T0 =Ti; T1=Pi; T2=To; T3=To
    elif si ==0 and sj == 1:
        T0 =Ti; T1=To; T2=To; T3=Pi
    elif si ==1 and sj == 1:
        T0 =Ti; T1=To; T2=Pi; T3=To

    return T0,T1,T2,T3
#==================================================================#


#==================================================================#
def eig_all (theta):
    try:
        Y, Z = linalg.eigh(theta)
    except linalg.linalg.LinAlgError:
        Y, Z = linalg.eig(theta)
    piv = np.argsort(Y)[::-1]
    Y = np.sqrt(np.abs(Y[piv]))
    Z = np.conj(Z[:,piv].T)
    return Y,Z
#==================================================================#


#==================================================================#
def merge_two(T1,T2,UU,UUT):

    D1 = UU.shape[0]; D2=int(np.sqrt(UU.shape[1]))
    UU = np.reshape(UU,(D1,D2,D2))
    UUT = np.reshape(UUT,(D2,D2,D1))

    Aup =  np.tensordot(T1,UU, axes=(2,1))
    Adown =  np.tensordot(T2,UUT, axes=(0,1))
    AO = np.tensordot(Aup,Adown,axes=([0,1,4],[3,2,1]) )
    AO = np.transpose(AO, (3,2,1,0))
    return AO
#==================================================================#




##
##          3             3
##     2--(TO)--0     2--(T1)--0
##          1              1
##
##          3             3
##     2--(T3)--0     2--(T2)--0
##          1             1



#==================================================================#
def updat_pure( T0,T3,Bond,dcut):

    if Bond=='x':
        T0 = np.transpose( T0,(1,0,3,2) )
        T3 = np.transpose( T3,(1,0,3,2) )

    dimT0 = T0.shape
    dimT3 = T3.shape

    Aup = np.tensordot( T0, T0.conjugate(), axes= ([2,3],[2,3]))
    Adown= np.tensordot( T3, T3.conjugate(), axes= ([2,1],[2,1]))
    AA =  np.tensordot( Aup, Adown, axes= ([1,3],[1,3]))
    AA =  np.reshape( np.transpose (AA,(0,2,1,3)),(dimT0[0]*dimT3[0],dimT0[0]*dimT3[0]))

    Yo,Zo = eig_all(AA)
    dc1 = np.min([np.sum(Yo>0),dcut])
    # dc1 = np.min([np.sum(Yo>10.**(-10)),dcut])
    UU = Zo [0:dc1,:];  UUT =(Zo.T[:,0:dc1]).conj();

    AO = merge_two(T0,T3,UU,UUT); N1 = np.max(abs(AO));
    NewT = AO/N1

    if Bond=='x': np.transpose(NewT,(1,0,3,2))

    return NewT,UU,UUT,N1
#==================================================================#
def updat_pure2( T0,T3,Bond,dcut):

    if Bond=='x':
        T0 = np.transpose( T0,(1,0,3,2) )
        T3 = np.transpose( T3,(1,0,3,2) )

    dimT0 = T0.shape
    dimT3 = T3.shape

    Aup = np.tensordot( T0, T0.conjugate(), axes= ([2,3],[2,3]))
    Adown= np.tensordot( T3, T3.conjugate(), axes= ([2,1],[2,1]))
    AA =  np.tensordot( Aup, Adown, axes= ([1,3],[1,3]))
    AA =  np.reshape( np.transpose (AA,(0,2,1,3)),(dimT0[0]*dimT3[0],dimT0[0]*dimT3[0]))

    Yo,Zo = eig_all(AA)

    dc1 = np.min([np.sum(Yo>0),dcut])
    # dc1 = np.min([np.sum(Yo>10.**(-10)),dcut])
    UU = UUT= np.identity(dc1)
    AO = merge_two(T0,T3,UU,UUT); N1 = np.max(abs(AO));
    NewT = AO

    if Bond=='x': np.transpose(NewT,(1,0,3,2))

    return NewT,UU,UUT,N1



#==================================================================#
def updat_impurity(T0,T3,Bond,dcut,UU,UUT,N1):

    if Bond=='x':
        T0 = np.transpose( T0,(1,0,3,2) )
        T3 = np.transpose( T3,(1,0,3,2) )

    AO = merge_two(T0,T3,UU,UUT);
    NewT = AO/N1

    if Bond=='x': np.transpose(NewT,(1,0,3,2))

    return NewT
#==================================================================#


#==================================================================#
def merge_four(TN):

    T03 =  np.tensordot(TN[0],TN[3], axes=([1,3],[3,1]))
    T12 =  np.tensordot(TN[1],TN[2], axes=([1,3],[3,1]))
    AO = np.tensordot( T03,T12,axes=([0,1,2,3],[1,0,3,2]) )

    return AO
#==================================================================#

def creatMemo(trgstep,sites=None):
    ''' create dict: site --> contraction graph (2X2), besides
        (i,j), there are two different keys, one is 'one side', which represent
        impurity at (0,0) . Another is 'denominator', that is, the pure tensor'''
    memo = {}
    memo['one side'] = [1 for _ in range(trgstep+1)]
    if sites != None:
        for site in sites:
            i,j = site
            memo[(i,j)] = site_graph_map(i,j,trgstep)
    else:
        for i in range(2**trgstep*2):
            for j in range(i,2**trgstep*2):
                memo[(i,j)] = site_graph_map(i,j,trgstep)
    memo['denominator'] = [8 for _ in range(trgstep+1)]
    return memo

def site_graph_map(i,j,trgsteps):
    result = []
    for trgstep in range(trgsteps+1):
        Lsi,si = divmod(i,2)
        Lsj,sj = divmod(j,2)
        if Lsi != 0 or Lsj != 0:  # two impurity have not met each other
            if si == 0 and sj == 0:
                result.append(1)
            elif si == 1 and sj == 0:
                result.append(2)
            elif si == 1 and sj == 1:
                result.append(3)
            elif si == 0 and sj == 1:
                result.append(4)
        else:    # two different cases, one is two sites have merged,
            if i ==0 and j == 0:
                result.append(1)
            elif si == 1 and sj == 0:
                result.append(5)
            elif si == 1 and sj == 1:
                result.append(6)
            elif si== 0 and sj == 1:
                result.append(7)
        i = Lsi
        j = Lsj
    assert(len(result) == trgsteps+1)
    return result

def graph_tensor_map(graph_num,D,Ti,Pi):
    ''' D : pure tensor; Ti: one side (top left) ; Pi: purity on (i,j) '''
    if graph_num == 8:
        T0 = T1 = T2 = T3 = D
    elif graph_num == 1:
        T0 = Pi; T1=T2=T3 = D
    elif graph_num == 2:
        T1 = Pi; T0=T2=T3 = D
    elif graph_num == 3:
        T2 = Pi; T0=T1=T3 = D
    elif graph_num == 4:
        T3 = Pi; T0=T1=T2 = D
    elif graph_num == 5:
        T0 = Ti;T1 = Pi; T2=T3 = D
    elif graph_num == 6:
        T0 = Ti; T2=Pi; T1=T3 = D
    elif graph_num == 7:
        T0=Ti; T3=Pi; T1=T2 = D
    return T0,T1,T2,T3
#==================================================================#
def Cmax_CHmax( DT,IDT,dcut):

    ## update along y-direction
    DT1,UU,UUT,N1 = updat_pure( DT,DT,'y',dcut)
    IDY1 = updat_impurity( IDT, DT,'y',dcut,UU,UUT,N1)
    IDY2 = updat_impurity(  DT,IDT,'y',dcut,UU,UUT,N1)


    ## update along x-direction
    DT2,UU,UUT,N1 = updat_pure( DT1,DT1,'x',dcut)
    ID1 = updat_impurity( IDY1, DT1,'x',dcut,UU,UUT,N1)
    ID2 = updat_impurity( IDY1,IDY2,'x',dcut,UU,UUT,N1)
    IX2 = updat_impurity( IDY1,IDY1,'x',dcut,UU,UUT,N1)

    Norm = merge_four(  [DT2,DT2,DT2,DT2] )

    Cm_D = merge_four([ID1, DT2, ID1, DT2])/Norm
    Ch_D = merge_four([ID2, DT2, DT2, DT2])/Norm
    Cm_X = merge_four([ID1, ID1, DT2, DT2])/Norm
    Ch_X = merge_four([IX2, DT2, DT2, DT2])/Norm

    return Cm_D,Ch_D,Cm_X,Ch_X
#==================================================================#



#==================================================================#
def M4_calculation (S0,S1,S2,S3,S4,dcut):


    for Bond in [ 'y', 'x']:

        nS0,UU,UUT,N1 = updat_pure( S0,S0,Bond,dcut)

        nS1 = 1./2.*( updat_impurity( S1, S0,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S0, S1,Bond,dcut,UU,UUT,N1))

        nS2 = 1./4.*( updat_impurity( S2, S0,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S0, S2,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S1, S1,Bond,dcut,UU,UUT,N1)*2 )

        nS3 = 1./8.*( updat_impurity( S3, S0,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S2, S1,Bond,dcut,UU,UUT,N1)*3 +
                      updat_impurity( S1, S2,Bond,dcut,UU,UUT,N1)*3 +
                      updat_impurity( S0, S3,Bond,dcut,UU,UUT,N1) )

        nS4 = 1./16.*( updat_impurity( S4, S0,Bond,dcut,UU,UUT,N1) +
                       updat_impurity( S3, S1,Bond,dcut,UU,UUT,N1)*4 +
                       updat_impurity( S2, S2,Bond,dcut,UU,UUT,N1)*6 +
                       updat_impurity( S1, S3,Bond,dcut,UU,UUT,N1)*4 +
                       updat_impurity( S0, S4,Bond,dcut,UU,UUT,N1) )

        S0 = nS0;  S1 = nS1;  S2 = nS2;  S3 = nS3;   S4 = nS4;

    return S0,S1,S2,S3,S4
#==================================================================#




#==================================================================#
def Ising_square(T):


        Tax = np.ones((2,2))
        taix = []

        DT = np.zeros((2,2,2,2))
        iDT = np.zeros((2,2,2,2))

        for ii in range(2):
            DT [ii,ii,ii,ii] = 1.0
            iDT [ii,ii,ii,ii] = -1.0

        iDT [0,0,0,0] = 1

        Tax [0,0] = np.exp((1./T))
        Tax [0,1] = np.exp((-1./T))
        Tax [1,0] = np.exp((-1./T))
        Tax [1,1] = np.exp((1./T))


        Ya, Za = np.linalg.eigh(Tax)
        taix.append(np.tensordot(Za, np.diag(Ya**0.5),axes=(1,0)) )
        taix.append(np.tensordot(np.diag(Ya**0.5), (Za.conj().T) ,axes=(1,0)))


        DT =  np.tensordot( DT,taix[0],axes=(0,0))
        DT =  np.tensordot( DT,taix[0],axes=(0,0))
        DT =  np.tensordot( DT,taix[1],axes=(0,1))
        DT =  np.tensordot( DT,taix[1],axes=(0,1))

#        print '================='

        iDT =  np.tensordot(iDT,taix[0],axes=(0,0))
        iDT =  np.tensordot(iDT,taix[0],axes=(0,0))
        iDT =  np.tensordot(iDT,taix[1],axes=(0,1))
        iDT =  np.tensordot(iDT,taix[1],axes=(0,1))

        return DT, iDT
#==================================================================#

#==================================================================#
def Ising_exact(N,beta,dpp):

    Ten= np.zeros((N,N,N,N), dtype=dpp)
    iTen= np.zeros((N,N,N,N), dtype=dpp)

    for ii in range(N**4):
        Lab = dNb(N,ii,4)
        if np.mod(np.sum(Lab),N)==0:
            N0 = 0
            for jj in range(len(Lab)):
                if Lab[jj]==0: N0+=1

            NL = 4-N0
            VAL =np.exp(-2*beta)*( (np.sqrt(np.exp(beta*2.)+N-1))**(N0)*\
                (np.sqrt(np.exp(beta*2.)-1))**(NL) )/N

            Ten[Lab[0],Lab[1],Lab[2],Lab[3]]= VAL

        if np.mod(np.sum(Lab),N)==1:
            N0 = 0
            for jj in range(len(Lab)):
                if Lab[jj]==0: N0+=1

            NL = 4-N0
            VAL =np.exp(-2*beta)*( (np.sqrt(np.exp(beta*2.)+N-1))**(N0)*\
                (np.sqrt(np.exp(beta*2.)-1))**(NL) )/N

            iTen[Lab[0],Lab[1],Lab[2],Lab[3]]= VAL

    return Ten, iTen
#==================================================================#

#==================================================================#
def dNb(number,n,L):
    bb=[ ]
    for i in range(L-1,-1,-1):
        ii=number**i
        x,y= divmod(n, ii);n=y
        bb.append( x )
    return np.array(bb,dtype=int)
#==================================================================#
