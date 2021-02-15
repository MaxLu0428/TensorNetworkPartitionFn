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
    except linalg.LinAlgError:
        Y, Z = linalg.eig(theta)
    piv = np.argsort(Y)[::-1]
    Y = np.sqrt(np.abs(Y[piv]))
    Z = np.conj(Z[:,piv].T)
    return Y,Z
#==================================================================#


#==================================================================#
def merge_two(T1,T2,UU,UUT):
    dimT1 = T1.shape
    DT1 = dimT1[0]
    dimT2 = T2.shape
    DT2 = dimT2[0]
    D1 = UU.shape[0];
    UU = np.reshape(UU,(D1,DT1,DT2))
    UUT = np.reshape(UUT,(DT1,DT2,D1))
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
    dc1 = np.min([len(Yo),dcut])
    # dc1 = np.min([np.sum(Yo>10.**(-10)),dcut])
    UU = Zo [0:dc1,:];  UUT =(Zo.T[:,0:dc1]).conj();

    AO = merge_two(T0,T3,UU,UUT); N1 = np.max(abs(AO));
    NewT = AO/N1

    if Bond=='x': NewT = np.transpose(NewT,(1,0,3,2))

    return NewT,UU,UUT,N1
#==================================================================#



#==================================================================#
def updat_impurity(T0,T3,Bond,dcut,UU,UUT,N1):

    if Bond=='x':
        T0 = np.transpose( T0,(1,0,3,2) )
        T3 = np.transpose( T3,(1,0,3,2) )

    AO = merge_two(T0,T3,UU,UUT);
    NewT = AO/N1

    if Bond=='x': NewT = np.transpose(NewT,(1,0,3,2))

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

def creatMemoForSy(trgstep,site=None):
    sites = [(x,y) for x in range(2**trgstep*2) for y in range(2**trgstep*2)]
    memo  =  creatMemo(trgstep,sites)
    for site in memo:
        # if another impurity in the same block
        if memo[site][0] == 1:
            memo[site][0] = 7
        elif memo[site][0] == 2:
            memo[site][0] = 9
    # prepare for another impurity at another block
    memo['aux1'] = [1 for _ in range(trgstep+1)]
    memo['aux2'] = [2] + [1 for _ in range(trgstep)]
    for site in memo:
        for i in memo[site][:-1]:
            if memo[site][i] == 4 and memo[site][i+1] == 7 and i != len(memo[site]-2):
                memo[site][i+2] = 10
            elif memo[site][i] == 4 and memo[site][i+1] == 1:
                memo[site][1] = 11
            elif memo[site][0] == 4 and memo[site][1] == 2:
                pass
            elif memo[site] == 3 and memo[site][1] == 7:
                memo[site][2] = 12
            elif memo[site] == 3 and memo[site][1] == 2 :
                memo[site][1] = 13

    memo[(1,0)] = [14] + [1 for _ in range(trgstep)]
    memo[(1,1)][1] = None






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


def correlation_lengh(S0,S1_q1,S2_q1,S1_q2,S2_q2,dcut,q1=(2*np.pi/(2**6),0),RG_step=0):
    q1 = np.array(q1)
    q2 = 2 * q1
    for Bond in [ 'y', 'x']:
        if Bond == 'y':
            T = np.array([0,2**RG_step])
        elif Bond == 'x':
            T = np.array([2**RG_step,0])

        nS0,UU,UUT,N1 = updat_pure( S0,S0,Bond,dcut)
        j = np.array(1j)
        phase1 = np.dot(T,q1)
        S1_q1_conj = np.conj(S1_q1)
        phase2 = np.dot(T,q2)
        S1_q2_conj = np.conj(S1_q2)
        nS1_q1 = 1./2.*(updat_impurity( S1_q1, S0,Bond,dcut,UU,UUT,N1)+
                        np.exp(j*phase1)*updat_impurity( S0, S1_q1,Bond,dcut,UU,UUT,N1))
        q1_cross = np.exp(j*phase1)*updat_impurity(S1_q1, S1_q1_conj,Bond,dcut,UU,UUT,N1)
        nS2_q1 = 1./4.*( updat_impurity( S2_q1, S0,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S0, S2_q1,Bond,dcut,UU,UUT,N1) +
                        q1_cross+ q1_cross.conj()
                       )
        nS1_q2 = 1./2.*(updat_impurity( S1_q2, S0,Bond,dcut,UU,UUT,N1)+
                        np.exp(j*phase2)*updat_impurity( S0, S1_q2,Bond,dcut,UU,UUT,N1))

        q2_cross = np.exp(j*phase2)*updat_impurity( S1_q2_conj, S1_q2,Bond,dcut,UU,UUT,N1)

        nS2_q2 = 1./4.*( updat_impurity( S2_q2, S0,Bond,dcut,UU,UUT,N1) +
                         updat_impurity( S0, S2_q2,Bond,dcut,UU,UUT,N1) +
                        q2_cross+ q2_cross.conj()
                       )

        S0 = nS0;  S1_q1 = nS1_q1;  S2_q1 = nS2_q1;  S1_q2 = nS1_q2;   S2_q2 = nS2_q2;
    return S0,S1_q1,S2_q1,S1_q2,S2_q2
#==================================================================#
def correlation_lengh2(S0,S1_q1,S2_q1,S1_q2,S2_q2,dcut,q1=(2*np.pi/(2**6),0),RG_step=0):
    q1 = np.array(q1)
    q2 = 2 * q1
    for Bond in [ 'y', 'x']:
        if Bond == 'y':
            T = np.array([0,2**RG_step])
        elif Bond == 'x':
            T = np.array([2**RG_step,0])

        nS0,UU,UUT,N1 = updat_pure( S0,S0,Bond,dcut)
        j = np.array(1j)
        phase1 = np.dot(T,q1)
        S1_q1_conj = np.conj(S1_q1)
        phase2 = np.dot(T,q2)
        S1_q2_conj = np.conj(S1_q2)
        nS1_q1 = 1./2.*(updat_impurity( S1_q1, S0,Bond,dcut,UU,UUT,N1)+
                        np.exp(j*phase1)*updat_impurity( S0, S1_q1,Bond,dcut,UU,UUT,N1))
        q1_cross = np.exp(j*phase1)*updat_impurity(S1_q1, S1_q1_conj,Bond,dcut,UU,UUT,N1)
        q1_cross_conj = np.exp(-j*phase1) * updat_impurity(S1_q1, S1_q1_conj,Bond,dcut,UU,UUT,N1)
        nS2_q1 = 1./4.*( updat_impurity( S2_q1, S0,Bond,dcut,UU,UUT,N1) +
                      updat_impurity( S0, S2_q1,Bond,dcut,UU,UUT,N1) +
                        q1_cross+ q1_cross_conj
                       )
        nS1_q2 = 1./2.*(updat_impurity( S1_q2, S0,Bond,dcut,UU,UUT,N1)+
                        np.exp(j*phase2)*updat_impurity( S0, S1_q2,Bond,dcut,UU,UUT,N1))

        q2_cross = np.exp(j*phase2)*updat_impurity( S1_q2_conj, S1_q2,Bond,dcut,UU,UUT,N1)
        q2_cross_conj = np.exp(-j*phase2) * updat_impurity(S1_q2, S1_q2_conj,Bond,dcut,UU,UUT,N1)
        nS2_q2 = 1./4.*( updat_impurity( S2_q2, S0,Bond,dcut,UU,UUT,N1) +
                         updat_impurity( S0, S2_q2,Bond,dcut,UU,UUT,N1) +
                        q2_cross+ q2_cross_conj
                       )

        S0 = nS0;  S1_q1 = nS1_q1;  S2_q1 = nS2_q1;  S1_q2 = nS1_q2;   S2_q2 = nS2_q2;
    return S0,S1_q1,S2_q1,S1_q2,S2_q2



#==================================================================#
def Ising_square(T,bias=10**-4):


        Tax = np.ones((2,2))
        taix = []

        DT = np.zeros((2,2,2,2))
        iDT = np.zeros((2,2,2,2))

        for ii in range(2):
            DT [ii,ii,ii,ii] = 1.0
            iDT [ii,ii,ii,ii] = -1.0
        DT[0,0,0,0] += bias
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
