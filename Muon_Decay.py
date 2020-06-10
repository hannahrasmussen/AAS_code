#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numba as nb


# Physical Constants:

# In[2]:


a_value = 1/137    #The fine structure constant; unitless
Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
me = .511          #Mass of an electron in MeV
mpi_neutral = 135  #Mass of a neutral pion in MeV
mpi_charged = 139.569  #Mass of a charged pion in MeV
mPL = 1.124*10**22 #Planck mass in MeV
mu = 105.661       #Mass of a muon in MeV
f_pi = 131         #MeV, not really sure what this constant means 
eps_e = me/mu      #constant used in eveyr integration
We = .5*mu*(1+(eps_e**2))     #constant used in electron/positron integration
x0e = 2*eps_e/(1+eps_e**2)    #constant used in electron/positron integration


# In[3]:


@nb.jit(nopython=True)
def decay7(ms,angle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(angle))**2
    gamma = part1*part2
    return 2*gamma #because vs can decay into either pi+ and u- OR pi- and u+


# In[4]:


@nb.jit(nopython=True)
def gammanu(Emax): #calculates gamma sub mu for both electron neutrinos and muon neutrinos
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_a1 = 3*(me**4)*(mu**2)*np.log(abs(2*Emax-mu))
    part_a2 = 6*(me**4)*Emax*(mu+Emax)
    part_a3 = 16*(me**2)*mu*(Emax**3)
    part_a4 = 4*(mu**2)*(Emax**3)*(3*Emax - 2*mu)
    part_a5 = 24*mu**3
    part_b1 = 3*(me**4)*(mu**2)*np.log(abs(-mu))/part_a5
    integral = ((part_a1+part_a2+part_a3+part_a4)/part_a5)-part_b1
    return -1*constant*integral


# In[5]:


@nb.jit(nopython=True)
def gammanu2(Emax,a,b): #b is the upper limit of integration and a is the lower limit; calculates dgamma/(Ev*dEv) for electron neutrinos and muon neutrinos
    if a>Emax:
        return 0
    constant = 8*Gf*(mu**2)/(16*np.pi**3)
    part_b1 = (-1/4)*(me**4)*mu*np.log(abs(2*b-mu))
    part_b2 = (-1/6)*b
    part_b3 = 3*(me**4)+6*(me**2)*mu*b
    part_b4 = (mu**2)*b*(4*b-3*mu)
    part_b = (part_b1+part_b2*(part_b3+part_b4))/mu**3
    part_a1 = (-1/4)*(me**4)*mu*np.log(abs(2*a-mu))
    part_a2 = (-1/6)*a
    part_a3 = 3*(me**4)+6*(me**2)*mu*a
    part_a4 = (mu**2)*a*(4*a-3*mu)
    part_a = (part_a1+part_a2*(part_a3+part_a4))/mu**3
    integral = part_b-part_a
    return constant*integral


# I'm thinking I'm going to make separate functions for the muon neutrino, electron neutrino, and electron (and their antiparticles?) so that each can just be called as needed and it shouldn't be too hard to sort out which particle we're talking about.

# In[6]:


@nb.jit(nopython=True)
def v(Eu,E,ms,angle): 
    #Eu is the energy of the decaying muon, which should be constant for a given ms
    #E is the hypothetical energy of the active neutrino in question
    #ms is the mass of the sterile neutrino
    #angle is the mixing angle of the sterile neutrino with the active neutrino flavors
    #This function will return dP/(dtdE) for the muon neutrino and the electron neutrino
    gammaL = Eu/mu
    pu = (Eu**2-mu**2)**(1/2)
    v = pu/Eu
    Emax = (mu/2)*(1-(eps_e)**2) #maximum possible energy of the active neutrino
    gam_nu = gammanu(Emax)
    gam_nu2 = gammanu2(Emax,E/(gammaL*(1+v)),min(Emax,E/(gammaL*(1-v))))
    return (decay7(ms,angle)/(2*gammaL*v))*(gam_nu2/gam_nu)


# In[7]:


@nb.jit(nopython=True)
def gammae(Emax): #calculates gamma sub mu for electrons and positrons
    constant = 2*Gf**2*We**4/(np.pi**3)
    part_a1 = We*(((Emax/We)**2-x0e**2)**(1/2))+Emax
    part_a2 = (1/8)*We*(x0e**4)*np.log(abs(part_a1))
    part_a3 = ((Emax/We)**2-x0e**2)**(1/2)
    part_a4 = -8*We**3*x0e**2 + 3*We**2*x0e**2*Emax + 8*We*Emax**2 - 6*Emax**3
    part_a5 = 24*We**2
    part_a = part_a2 + (part_a3)*(part_a4)/part_a5
    #part_b1 = We*((-x0e**2)**(1/2)) #this number is going to be imaginary
    #part_b2 = (1/8)*We*(x0e**4)*np.log(abs(part_b1))
    #part_b3 = (-x0e**2)**(1/2) #this number will also be imaginary
    #part_b4 = -8*We**3*x0e**2 
    #part_b5 = 24*We**2
    #part_b = part_b2 + (part_b3)*(part_b4)/part_b5
    return constant*part_a


# In[8]:


@nb.jit(nopython=True)
def gammae2(a,b): #calculates dgamma/(Ev*dEv) for electrons and positrons; b is upper limit of integration, a is lower limit
    constant = 2*Gf**2*We**4/(np.pi**3)
    part_b1 = 2*We**2*x0e**2 + 3*We*b - 2*b**2
    part_b2 = ((b/We)**2-x0e**2)**(1/2)
    part_b3 = 6*We**2
    part_b4 = We*(((b/We)**2-x0e**2)**(1/2))+b
    part_b5 = .5*(x0e**2)*np.log(part_b4)
    part_b = part_b1*part_b2/part_b3 - part_b5
    part_a1 = 2*We**2*x0e**2 + 3*We*a - 2*a**2
    part_a2 = ((a/We)**2-x0e**2)**(1/2)
    part_a3 = 6*We**2
    part_a4 = We*(((a/We)**2-x0e**2)**(1/2))+a
    part_a5 = .5*(x0e**2)*np.log(part_a4)
    part_a = part_a1*part_a2/part_a3 - part_a5
    integral = part_b-part_a
    return constant*integral


# In[9]:


@nb.jit(nopython=True)
def e(Eu,ms,angle):
    #Eu is the energy of the decaying muon
    #ms is the mass of the sterile neutrino
    #angle is the mixing angle of the sterile neutrino with the active neutrino flavors
    #This function will return dP/(dtdE) for the electron, but we're not really using this anywhere yet
    gammaL = Eu/mu
    pu = (Eu**2-mu**2)**(1/2)
    v = pu/Eu
    Emax = We
    gam_e = gammae(Emax)
    gam_e2 = gammae2(Eu/(gammaL*(1+v)),min(Emax,Eu/(gammaL*(1-v))))
    return (decay7(ms,angle)/(2*gammaL*v))*(gam_e2/gam_e)

