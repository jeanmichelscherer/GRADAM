#!/usr/bin/python
import numpy as np

def Ctensor(ndim,E,nu):

    sh=shear(E,nu)   
    lm=lame(E,nu)
    if ndim==2:
        C1=np.ndarray([2,2,2,2])
        C1[0][0][0][0]=lm+2*sh
        C1[0][0][1][1]=lm
        C1[0][0][1][0]=0.
        C1[0][0][0][1]=0.
        C1[1][1][0][0]=lm
        C1[1][1][1][1]=lm+2*sh
        C1[1][1][1][0]=0.
        C1[1][1][0][1]=0.
        C1[0][1][0][0]=0.
        C1[0][1][1][1]=0.
        C1[0][1][0][1]=sh
        C1[0][1][1][0]=sh
        C1[1][0][0][0]=0.
        C1[1][0][1][1]=0.
        C1[1][0][0][1]=sh
        C1[1][0][1][0]=sh
    if ndim==3:
        C1=np.zeros([3,3,3,3])
        C1[0][0][0][0]=lm+2*sh
        C1[1][1][1][1]=lm+2*sh
        C1[2][2][2][2]=lm+2*sh
        C1[0][0][1][1]=lm
        C1[0][0][2][2]=lm
        C1[1][1][0][0]=lm
        C1[1][1][2][2]=lm
        C1[2][2][0][0]=lm
        C1[2][2][1][1]=lm
#        C1[0][0][0][0]=lm+4*sh/3
#        C1[1][1][1][1]=lm+4*sh/3
#        C1[2][2][2][2]=lm+4*sh/3
#        C1[0][0][1][1]=lm-2*sh/3
#        C1[0][0][2][2]=lm-2*sh/3
#        C1[1][1][0][0]=lm-2*sh/3
#        C1[1][1][2][2]=lm-2*sh/3
#        C1[2][2][0][0]=lm-2*sh/3
#        C1[2][2][1][1]=lm-2*sh/3
        C1[0][1][0][1]=sh
        C1[1][0][1][0]=sh
        C1[0][1][1][0]=sh
        C1[1][0][0][1]=sh
        C1[0][2][0][2]=sh
        C1[2][0][2][0]=sh
        C1[0][2][2][0]=sh
        C1[2][0][0][2]=sh
        C1[1][2][1][2]=sh
        C1[2][1][2][1]=sh
        C1[1][2][2][1]=sh
        C1[2][1][1][2]=sh
    return C1

def Ctensor_anis(ndim,R,**kwargs):
   if(ndim==2):
       c11=kwargs.get('c11')
       c12=kwargs.get('c12')
       c44=kwargs.get('c44')
       if(c44==None):
           c44=(c11-c12)/2.
       c22=kwargs.get('c22')
       if(c22==None):
           c22=c11
       C1=np.zeros([2,2,2,2])
       C1[0][0][0][0]=c11
       C1[0][0][1][1]=c12
       C1[1][1][0][0]=c12
       C1[1][1][1][1]=c22
       C1[0][1][0][1]=c44
       C1[0][1][1][0]=c44
       C1[1][0][0][1]=c44
       C1[1][0][1][0]=c44
       return np.einsum('mi,nj,pk,ql,mnpq->ijkl',R,R,R,R,C1)
   else:
       c11=kwargs.get('c11')
       c22=kwargs.get('c22')
       if(c22==None):
           c22=c11
       c33=kwargs.get('c33')
       if(c33==None):
           c33=c22
       c12=kwargs.get('c12')
       c13=kwargs.get('c13')
       if(c13==None):
           c13=c12
       c23=kwargs.get('c23')
       if(c23==None):
           c23=c12
       c44=kwargs.get('c44')
       if(c44==None):
           c44=(c11-c12)/2.
       c55=kwargs.get('c55')
       if(c55==None):
           c55=(c11-c13)/2.
       c66=kwargs.get('c66')
       if(c66==None):
           c66=(c22-c23)/2.
       C1=np.zeros([3,3,3,3])
       C1[0][0][0][0]=c11
       C1[1][1][1][1]=c22
       C1[2][2][2][2]=c33
       C1[0][0][1][1]=c12
       C1[0][0][2][2]=c13
       C1[1][1][0][0]=c12
       C1[1][1][2][2]=c23
       C1[2][2][0][0]=c13
       C1[2][2][1][1]=c23
       C1[0][1][0][1]=c44
       C1[1][0][1][0]=c44
       C1[0][1][1][0]=c44
       C1[1][0][0][1]=c44
       C1[0][2][0][2]=c55
       C1[2][0][2][0]=c55
       C1[0][2][2][0]=c55
       C1[2][0][0][2]=c55
       C1[1][2][1][2]=c66
       C1[2][1][2][1]=c66
       C1[1][2][2][1]=c66
       C1[2][1][1][2]=c66
       return np.einsum('mi,nj,pk,ql,mnpq->ijkl',R,R,R,R,C1)

def elastic(ndim,epsilon,E,nu,**kwargs):
    import numpy as np
    if E!=None and nu!=None:
        sh=shear(E,nu)   
        lm=lame(E,nu)
    C1=kwargs.get('C')
    if C1 is None:
        if ndim==2:
            C1=np.zeros([2,2,2,2])
            C1[0][0][0][0]=lm+2*sh
            C1[0][0][1][1]=lm
            C1[0][0][1][0]=0.
            C1[0][0][0][1]=0.
            C1[1][1][0][0]=lm
            C1[1][1][1][1]=lm+2*sh
            C1[1][1][1][0]=0.
            C1[1][1][0][1]=0.
            C1[0][1][0][0]=0.
            C1[0][1][1][1]=0.
            C1[0][1][0][1]=sh
            C1[0][1][1][0]=sh
            C1[1][0][0][0]=0.
            C1[1][0][1][1]=0.
            C1[1][0][0][1]=sh
            C1[1][0][1][0]=sh  
        if ndim==3:
            C1=np.zeros([3,3,3,3])
            C1[0][0][0][0]=lm+2*sh
            C1[1][1][1][1]=lm+2*sh
            C1[2][2][2][2]=lm+2*sh
            C1[0][0][1][1]=lm
            C1[0][0][2][2]=lm
            C1[1][1][0][0]=lm
            C1[1][1][2][2]=lm
            C1[2][2][0][0]=lm
            C1[2][2][1][1]=lm
            C1[0][1][0][1]=sh
            C1[1][0][1][0]=sh
            C1[0][1][1][0]=sh
            C1[1][0][0][1]=sh
            C1[0][2][0][2]=sh
            C1[2][0][2][0]=sh
            C1[0][2][2][0]=sh
            C1[2][0][0][2]=sh
            C1[1][2][1][2]=sh
            C1[2][1][2][1]=sh
            C1[1][2][2][1]=sh
            C1[2][1][1][2]=sh
    sigma=np.einsum('ijkl,kl->ij',C1,epsilon)
    return sigma,C1
   
def shear(E,nu):
    return E/(2.*(1+nu))

def lame(E,nu):
    return E*nu/((1+nu)*(1.-2*nu))
    
def young(sh,lm):
    return sh*(3*lm+2*sh)/(lm+sh)
    
def poisson(sh,lm):
    return sh/(2*(sh+lm))

def bulk(E,nu):
    return E/(3*(1-2*nu))
    