import numpy as np
import cytnx as cy

def mappingDtype2Cytnx(dpp):
    mapping = {'float':cy.Type.Float,
               'int':cy.Type.Int32,
               'complex':cy.Type.ComplexFloat,
               'bool':cy.Type.Bool}
    try:
        return mapping[dpp]
    except KeyError:
        print("no key {} inside mapping dtype function".format(dpp))

def Ising_exact(N,beta,dpp):
    CyDtype = mappingDtype2Cytnx(dpp)
    Ten = cy.zeros((N,N,N,N),dtype=CyDtype)
    iTen = cy.zeros((N,N,N,N),dtype=CyDtype)
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

def Ising_exact2(N,beta,dpp,weightMatrix=None):
    """ N: vertex of each node ; beta: J/(KB*T),dpp: datatype --> tensor Ten/iTen,
    try to construct the whole partition function with tensor Ten and iTen(impurity).
    For Ising model the weight is given by (assume kB=J=1)
            [ [exp(beta),exp(-beta)],
              [exp(-beta),exp(beta)]]
    """
    if not weightMatrix : # default will be Ising weight
        weightMatrix = np.array([[np.exp(beta),np.exp(-1*beta)],
                                 [np.exp(-1*beta),np.exp(beta)]])
        weightMatrix = cy.form_numpy(weightMatrix)
    CyDtype = mappingDtype2Cytnx(dpp)
    NetworkConstruct = PartitionNetwork('square')
    pureNode = cy.zeros((N,N,N,N),dtype=CyDtype)
    pureNode[0,0,0,0] = pureNode[1,1,1,1] = 1
    impureNode = cy.zeros((N,N,N,N),dtype=CyDtype)
    impureNode[0,0,0,0] = 1;impureNode[1,1,1,1] = -1
    Ten = NetworkConstruct(pureNode,weightMatrix).next()
    iTen = NetworkConstruct(impureNode,weightMatrix).next()
    return Ten, iTen

def dNb(number,n,L):
    bb=[ ]
    for i in range(L-1,-1,-1):
        ii=number**i
        x,y= divmod(n, ii);n=y
        bb.append( x )
    return np.array(bb,dtype=int)

def updat_pure(T0,T3,Bond,dcut):
    if Bond == "x":
        T0 = T0.permute([1,0,3,2])
        T0.set_labels([0,1,2,3])
        T3 = T3.permute([1,0,3,2])
        T3.set_labels([0,1,2,3])
    dimT0 = T0.shape()
    dimT3 = T3.shape()
#   contract T0/T0* and T3/T3* first ( for computation efficiency)
    T0_conj = T0.conj()
    T0.set_labels([0,1,-1,-2])
    T0_conj.set_labels([2,3,-1,-2])
    Aup = T0.contract(T0_conj)
    Aup = Aup.permute([0,1,2,3],rowrank=2)
    Aup.set_labels([0,-1,1,-2])
    # T3/T3*
    T3_conj = T3.conj()
    T3.set_labels([0,1,-1,-2])
    T3_conj.set_labels([2,3,-1,-2])
    Adown = T3.contract(T3_conj)
    Adown = Adown.permute([0,1,2,3],rowrank=2)
    Adown.set_labels([2,-1,3,-2])
    AA = Aup.contract(Adown)
