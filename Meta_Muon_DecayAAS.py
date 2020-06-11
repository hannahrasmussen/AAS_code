#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import random
import numba as nb


# In[2]:


a_value = 1/137    #The fine structure constant; unitless
Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
me = .511          #Mass of an electron in MeV
mpi_neutral = 135  #Mass of a neutral pion in MeV
mpi_charged = 139.569  #Mass of a charged pion in MeV
mPL = 1.124*10**22 #Planck mass in MeV
mu = 105.661       #Mass of a muon in MeV
f_pi = 131         #MeV, not really sure what this constant means 
eps_e = me/mu      #constant used in every integration
We = .5*mu*(1+(eps_e**2))     #constant used in electron/positron integration
x0e = 2*eps_e/(1+eps_e**2)    #constant used in electron/positron integration


# In[3]:


n = 10                        #number of steps for gauss laguerre and gauss legendre quadrature
num = 102                     #number of intervals for Fe plus two (for temp and time)
steps = 120                   #number of steps for DES
d_steps = 10                  #number of steps between steps for DES
a_init = 1/50                 #initial scale factor
a_final = 1200                #final goal scale factor
y_init = np.zeros(num)        
y_init[-2] = 50               #initial temperature in MeV
y_init[-1] = 0                #initial time
boxsize = 2
x_values, w_valuese = np.polynomial.legendre.leggauss(n)

for i in range (num-2):
    y_init[i] = 0

mixangle = .00005 #mixing angle of the sterile neutrino with the active neutrinos

#constants relating to the sterile neutrino:
D = 1           #a parameter that acts as a fraction of the number density of fermions?
fT = .5          #fraction of the mass converted to thermal energy in the plasma
mH = 300          #MeV, mass of sterile neutrino


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


# In[6]:


@nb.jit(nopython=True)
def f(Eu,E,ms,angle): 
    #Eu is the energy of the decaying muon, which should be constant for a given ms
    #E is the hypothetical energy of the active neutrino in question
    #ms is the mass of the sterile neutrino
    #angle is the mixing angle of the sterile neutrino with the active neutrino flavors
    #This function will return dP/(dtdE) for the muon neutrino and the electron neutrino
    gammau = Eu/mu
    pu = (Eu**2-mu**2)**(1/2)
    vu = pu/Eu
    Emax = (mu/2)*(1-(eps_e)**2) #maximum possible energy of the active neutrino
    gam_nu = gammanu(Emax)
    gam_nu2 = gammanu2(Emax,E/(gammau*(1+vu)),min(Emax,E/(gammau*(1-vu))))
    return (1/(2*gammau*vu))*gam_nu2/gam_nu


# In[7]:


@nb.jit(nopython=True)
def u_integral(E_mumin,E_mumax,Eactive,ms,angle):
    #E_mumin and E_mumax are the min and max energies of the muon from the pion decay
    #Eactive is the hypothetical energy of the active neutrino 
    #ms is the mass of the sterile neutrino
    #angle is the mixing angle of the sterile neutrino with the active neutrino flavors
    Eu_array = ((E_mumax-E_mumin)/2)*x_values + ((E_mumax+E_mumin)/2)
    integral = 0
    for i in range(n):
        integral = integral + (w_valuese[i]*((E_mumax-E_mumin)/2)*f(Eu_array[i],Eactive,ms,angle))
    return integral

