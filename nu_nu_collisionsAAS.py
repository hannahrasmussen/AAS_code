
# coding: utf-8

# In[38]:


import numpy as np

from numba import jit, prange



# In[39]:


@jit(nopython=True)
def J1(p1,p2,p3):
    return ((16/15)*p3**3*(10*(p1+p2)**2-15*p3*(p1+p2)+6*p3**2))

@jit(nopython=True)
def J2(p1,p2):
    return ((16/15)*p2**3*(10*p1**2+5*p1*p2+p2**2))

@jit(nopython=True)
def J3(p1,p2,p3):
    return ((16/15)*((p1+p2)**5-10*p3**3*(p1+p2)**2+15*p3**4*(p1+p2)-6*p3**5))



# In[40]:


@jit(nopython=True)
def b1(i,j,f1,p,dp):
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(j+1-h)
    w=0
    for q in range(h,j+1):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J1(p[i],p[j],p[int(k[q])])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J1(p[i],p[j],p[int(k[q])])))
    bpi=(1-f1[i])*(1-f1[j])*f1[h]*f1[u]*J1(p[i],p[j],p[h])
    bpf=(1-f1[i])*(1-f1[j])*f1[j]*f1[i]*J1(p[i],p[j],p[j])
    bni=f1[i]*f1[j]*(1-f1[h])*(1-f1[u])*J1(p[i],p[j],p[h])
    bnf=f1[i]*f1[j]*(1-f1[j])*(1-f1[i])*J1(p[i],p[j],p[j])
    return ((dp/2)*(bp-bpi-bpf),(dp/2)*(bn-bni-bnf))

@jit(nopython=True)
def A1(i,f1,p,dp):
    bP=np.zeros(i+1)
    bN=np.zeros(i+1)
    for j in range(i+1):
        bP[j],bN[j]=b1(i,j,f1,p,dp)
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an

@jit(nopython=True)
def b2(i,j,f1,p,dp):
    if j>=i:
        m=i
        n=j+1
        o=-1
    else:
        m=j
        n=i+1
        o=1
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(n-m)
    w=0
    for q in range(m,n):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J2(p[i],p[j])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J2(p[i],p[j])))
    bpi=(1-f1[i])*(1-f1[j])*f1[j]*f1[i]*J2(p[i],p[j])
    bpf=(1-f1[i])*(1-f1[j])*f1[i]*f1[j]*J2(p[i],p[j])
    bni=f1[i]*f1[j]*(1-f1[j])*(1-f1[i])*J2(p[i],p[j])
    bnf=f1[i]*f1[j]*(1-f1[i])*(1-f1[j])*J2(p[i],p[j])
    return ((dp/2)*(bp-bpi-bpf)*o,(dp/2)*(bn-bni-bnf)*o)

@jit(nopython=True)
def A2(i,f1,p,dp):
    bP=np.zeros(i+1)
    bN=np.zeros(i+1)
    for j in range(i+1):
        bP[j],bN[j]=b2(i,j,f1,p,dp)
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an

@jit(nopython=True)
def b3(i,j,f1,p,dp):
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(u+1-i)
    w=0
    for q in range(i,u+1):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J3(p[i],p[j],p[int(k[q])])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J3(p[i],p[j],p[int(k[q])])))
    bpi=(1-f1[i])*(1-f1[j])*f1[i]*f1[j]*J3(p[i],p[j],p[i])
    bpf=(1-f1[i])*(1-f1[j])*f1[u]*f1[h]*J3(p[i],p[j],p[u])
    bni=f1[i]*f1[j]*(1-f1[i])*(1-f1[j])*J3(p[i],p[j],p[i])
    bnf=f1[i]*f1[j]*(1-f1[u])*(1-f1[h])*J3(p[i],p[j],p[u])
    return ((dp/2)*(bp-bpi-bpf),(dp/2)*(bn-bni-bnf))

@jit(nopython=True)
def A3(i,f1,p,dp):
    bP=np.zeros(i+1)
    bN=np.zeros(i+1)
    for j in range(i+1):
        bP[j],bN[j]=b3(i,j,f1,p,dp)
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an

@jit(nopython=True)
def b4(i,j,f1,p,dp):
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(i+1-h)
    w=0
    for q in range(h,i+1):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J1(p[i],p[j],p[int(k[q])])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J1(p[i],p[j],p[int(k[q])])))
    bpi=(1-f1[i])*(1-f1[j])*f1[h]*f1[u]*J1(p[i],p[j],p[h])
    bpf=(1-f1[i])*(1-f1[j])*f1[i]*f1[j]*J1(p[i],p[j],p[i])
    bni=f1[i]*f1[j]*(1-f1[h])*(1-f1[u])*J1(p[i],p[j],p[h])
    bnf=f1[i]*f1[j]*(1-f1[i])*(1-f1[j])*J1(p[i],p[j],p[i])
    return ((dp/2)*(bp-bpi-bpf),(dp/2)*(bn-bni-bnf))

@jit(nopython=True)
def A4(i,f1,p,dp):
    bP=np.zeros(len(f1)-1-i)
    bN=np.zeros(len(f1)-1-i)
    v=0
    for j in range(i,len(f1)-1):
        bP[v],bN[v]=b4(i,j,f1,p,dp)
        v=v+1
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an

@jit(nopython=True)
def b5(i,j,f1,p,dp):
    if i>=j:
        m=j
        n=i+1
        o=-1
    else:
        m=i
        n=j+1
        o=1
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(n-m)
    w=0
    for q in range(m,n):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J2(p[j],p[i])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J2(p[j],p[i])))
    bpi=(1-f1[i])*(1-f1[j])*f1[i]*f1[j]*J2(p[j],p[i])
    bpf=(1-f1[i])*(1-f1[j])*f1[j]*f1[i]*J2(p[j],p[i])
    bni=f1[i]*f1[j]*(1-f1[i])*(1-f1[j])*J2(p[j],p[i])
    bnf=f1[i]*f1[j]*(1-f1[j])*(1-f1[i])*J2(p[j],p[i])
    return ((dp/2)*(bp-bpi-bpf)*o,(dp/2)*(bn-bni-bnf)*o)

@jit(nopython=True)
def A5(i,f1,p,dp):
    bP=np.zeros(len(f1)-1-i)
    bN=np.zeros(len(f1)-1-i)
    v=0
    for j in range(i,len(f1)-1):
        bP[v],bN[v]=b5(i,j,f1,p,dp)
        v=v+1
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an

@jit(nopython=True)
def b6(i,j,f1,p,dp):
    if i+j>=len(f1)-1:
        u=len(f1)-1
        h=(i+j)-u
    else:
        u=i+j
        h=0
    k=np.zeros(u+1-j)
    w=0
    for q in range(j,u+1):
        k[w]=q
        w=w+1
    bp=0
    bn=0
    for q in range(len(k)):
        bp=bp+(2*((1-f1[i])*(1-f1[j])*f1[int(k[q])]*f1[int(i+j-k[q])]*J3(p[i],p[j],p[int(k[q])])))
        bn=bn+(2*(f1[i]*f1[j]*(1-f1[int(k[q])])*(1-f1[int(i+j-k[q])])*J3(p[i],p[j],p[int(k[q])])))
    bpi=(1-f1[i])*(1-f1[j])*f1[j]*f1[i]*J3(p[i],p[j],p[j])
    bpf=(1-f1[i])*(1-f1[j])*f1[u]*f1[h]*J3(p[i],p[j],p[u])
    bni=f1[i]*f1[j]*(1-f1[j])*(1-f1[i])*J3(p[i],p[j],p[j])
    bnf=f1[i]*f1[j]*(1-f1[u])*(1-f1[h])*J3(p[i],p[j],p[u])
    return ((dp/2)*(bp-bpi-bpf),(dp/2)*(bn-bni-bnf))

@jit(nopython=True)
def A6(i,f1,p,dp):
    bP=np.zeros(len(f1)-1-i)
    bN=np.zeros(len(f1)-1-i)
    v=0
    for j in range(i,len(f1)-1):
        bP[v],bN[v]=b6(i,j,f1,p,dp)
        v=v+1
    ap=(dp/2)*(np.sum(2*bP)-bP[0]-bP[-1])
    an=(dp/2)*(np.sum(2*bN)-bN[0]-bN[-1])
    return ap,an



# In[44]:


GF=1.166*10**-11  #MeV**-2
@jit(nopython=True)
def cI(i,f1,p):
    dp=p[1]-p[0]
    Bleh=GF**2/((2*np.pi)**3*p[i]**2)
    AP1,AN1=A1(i,f1,p,dp)
    AP2,AN2=A2(i,f1,p,dp)
    AP3,AN3=A3(i,f1,p,dp)
    AP4,AN4=A4(i,f1,p,dp)
    AP5,AN5=A5(i,f1,p,dp)
    AP6,AN6=A6(i,f1,p,dp)
    c=Bleh*((AP1-AN1)+(AP2-AN2)+(AP3-AN3)+(AP4-AN4)+(AP5-AN5)+(AP6-AN6))
    FRS=Bleh*((AP1+AN1)+(AP2+AN2)+(AP3+AN3)+(AP4+AN4)+(AP5+AN5)+(AP6+AN6))
    return c,FRS

@jit(nopython=True,parallel=True)
def C(p,f1):
    c=np.zeros(len(p))
    FRS=np.zeros(len(p))
    for i in prange(1,len(p)-1): #i only goes up to 199 because in a# def's goes i-200 nd if i is 200 then len0
        c[i],FRS[i]=cI(i,f1,p)
        if (np.abs(c[i])/FRS[i])<=3e-15:
            c[i]=0
    return c

@jit(nopython=True)
def C_nopar(p,f1):
    c=np.zeros(len(p))
    FRS=np.zeros(len(p))
    for i in range(1,len(p)-1): #i only goes up to 199 because in a# def's goes i-200 nd if i is 200 then len0
        c[i],FRS[i]=cI(i,f1,p)
        if (np.abs(c[i])/FRS[i])<=3e-15:
            c[i]=0
    return c


