import numpy as np
from math import pi
from scipy import linalg
import itertools

#================================================#
def dNb(number,n,L):
    bb=[ ]
    for i in range(L-1,-1,-1):
        ii=number**i
        x,y= divmod(n, ii);n=y
        bb.append( x )
    return np.array(bb,dtype=int)


#================================================#
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














def transfer_matrix ( AA, a_label ):


    LEFT = []; LEFT_label = []
    for k in a_label:
        i = np.array(k,dtype=int)
        LEFT_label.append(i)
        LEFT.append( i )



    # Sort the basis
    nt_left_pos = {};
    for val in np.unique(LEFT ):
        nt_left_pos[val]  = np.where(LEFT == val)[0]


    Mat = []
    qun = [0,1]


    theta = np.trace( np.transpose( AA,(1,3,0,2))  )


    jj =-1
    for val, left_pos in nt_left_pos.items():
        jj +=1
        lval = qun[val];
        if lval in qun:
            left_pos = nt_left_pos[lval]
            beta = theta[np.ix_(left_pos,left_pos)]

            Y_block, Z_block = np.linalg.eig(beta)

            piv = np.argsort(abs(Y_block))[::-1]
            Y_block = Y_block[piv]
            Mat.append( np.real(Y_block) )

    return Mat




















def determine_U ( theta, a_label,chi_max ):

#    print (a_label)

    LEFT = []; LEFT_label = []
    for k in itertools.product(a_label, a_label ):

        i = np.array(k,dtype=int)
        i = np.reshape (np.array(i,dtype=int),(2))
        LEFT_label.append(i)
        LEFT.append(  np.mod(  (i[0] + i[1]), 2) )


    # Sort the basis
    nt_left_pos = {};
    for val in np.unique(LEFT ):
        nt_left_pos[val]  = np.where(LEFT == val)[0]


    qun = [0,1]



    dimTt = theta.shape

    Y = np.zeros(( dimTt[0],dimTt[1]))
    Z = np.zeros(( dimTt[0],dimTt[1]),dtype='complex')

    first_time = True;   lower_index = 0
    for val, left_pos in nt_left_pos.items():

        lval = qun[val]


        if lval in nt_left_pos:
            left_pos = nt_left_pos[lval]

            beta = theta[np.ix_(left_pos,left_pos)]

            Y_block, Z_block = np.linalg.eig(beta)

            if first_time:
                Y = Y_block
                n_left = left_pos
                first_time = False
            else:
                Y = np.append(Y,Y_block)
                n_left = np.append( n_left, left_pos)


            Z[left_pos,lower_index:lower_index+len(Y_block)] = Z_block
            lower_index += len(Y_block)


    if (lower_index < np.shape(Z)[0]):
        Y = np.append(Y, np.zeros(np.shape(Z)[0]-lower_index))
        n_left  = np.append(n_left,  n_left[-1]*np.ones(np.shape(Z)[0]-lower_index ,dtype='int'))


    # Obtain the new values
    chi2 =  np.min([np.sum(abs(Y)>10.**(-10)),chi_max])
    piv = np.argsort(abs(Y))[-1:-chi2-1:-1]
    L_label = n_left[piv]


    Z = Z.T[piv,:]

    UU = Z [0:chi2,:]




    # Given new quantum number
    aa_label= []
    for ii in L_label:
        ts = list( LEFT_label[ii][:] )

        t1 = np.mod ( (ts[0]+ts[1]),2)
        aa_label.append(t1)    ## new ( a,a' )


    return UU, aa_label











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
