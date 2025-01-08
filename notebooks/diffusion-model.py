#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# Here we develop a one-dimensional model of diffusion.
# It assumes a constant diffusivity.
# It useas a regular grid.
# It has fixed boundary conditions.

# The diffusion equation:
# 
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# 
# The discretized version of the diffusion equation that we'll solve with our model:
# 
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# 
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011). (Or see Wikipedia.)

# We will use two libraries, Numpy (for arrays)
# and Matplotlib (for plotting)
# that aren't a part of the base Python distribution.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Set two fixed model parameters, the diffusivity and the size of the model domain.

# In[23]:


D = 100 # diffusivity
Lx = 300 # domain size


# Next, set up the model grid using a NumPy array.

# In[10]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)


# In[11]:


whos


# In[12]:


x


# In[13]:


x[0]


# In[14]:


x[nx-1]


# In[15]:


x[-1]


# In[16]:


x[0:5]


# Set the initial concentration profile for the model.
# The concentration 'C' is a step function with a high value on the left,
# a low value on the right,
# and the step as the center of the domain.

# In[17]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C [x <= Lx//2] = C_left
C [x > Lx//2] = C_right


# Plot the initial profile.

# In[18]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")


# Set the start time of the model and the number of time steps. Calculate a stable time step for the model using a stability criterion.

# In[21]:


time = 0
nt =5000
dt = 0.5 * (dx**2 / D)


# In[22]:


dt


# In[24]:


z = 5


# In[25]:


z = z + 5
z


# In[26]:


z += 5
z


# Loop over the time steps of the model, solving the diffusion equation using the FTCS explicit scheme
# described above.
# The boundary conditions are fixed, so reset them at each time step.

# In[33]:


for t in range (0, nt):
    C += D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C, 1))
    C[0] = C_left
    C[-1] = C_right


# Plot the result. 

# In[35]:


plt.figure()
plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final concentration profile")


# In[ ]:




