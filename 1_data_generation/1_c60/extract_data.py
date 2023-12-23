#!/usr/bin/env python

# coding: utf-8

# In[2]:


import os
import time
import numpy as np
import libra_py.packages.cp2k.methods as CP2K_methods
#from IPython.display import clear_output


# In[ ]:


# First the atomic guesses
istep = 999
fstep = 3001
overlaps = []
coeffs = []
ks_mats = []
for step in range(istep,fstep):
    filename = F'2_atomic_wfn/c60_{step}-data-1_0.Log'
    data = CP2K_methods.read_ao_matrices(filename)
    filename = F'2_atomic_wfn/c60_{step}-RESTART.wfn'
    _, _, _, mo_coeffs = CP2K_methods.read_wfn_file(filename)
    overlaps.append(data[0])
    coeffs.append(mo_coeffs)
    ks_mats.append(data[1])
    #clear_output()
overlaps = np.array(overlaps)
coeffs = np.array(coeffs)
ks_mats = np.array(ks_mats)
print(ks_mats.shape, coeffs.shape, overlaps.shape)
np.save('c60_pbe_atomic_overlaps.npy', overlaps)
np.save('c60_pbe_atomic_ks_mats.npy', ks_mats)
np.save('c60_pbe_atomic_coeffs.npy', coeffs)

# Second the PBE converged wfns
istep = 999
fstep = 3001
overlaps = []
coeffs = []
ks_mats = []
for step in range(istep,fstep):
    filename = F'3_converged_wfn/c60_{step}-data-1_0.Log'
    data = CP2K_methods.read_ao_matrices(filename)
    filename = F'3_converged_wfn/c60_{step}-RESTART.wfn'
    _, _, _, mo_coeffs = CP2K_methods.read_wfn_file(filename)
    overlaps.append(data[0])
    coeffs.append(mo_coeffs)
    ks_mats.append(data[1])
    #clear_output()
overlaps = np.array(overlaps)
coeffs = np.array(coeffs)
ks_mats = np.array(ks_mats)
print(ks_mats.shape, coeffs.shape, overlaps.shape)
np.save('c60_pbe_converged_overlaps.npy', overlaps)
np.save('c60_pbe_converged_ks_mats.npy', ks_mats)
np.save('c60_pbe_converged_coeffs.npy', coeffs)

# Third the B3LYP converged wfns
istep = 999
fstep = 3001
overlaps = []
coeffs = []
ks_mats = []
for step in range(istep,fstep):
    print(step)
    filename = F'4_converged_wfn_b3lyp/c60_{step}-data-1_0.Log'
    data = CP2K_methods.read_ao_matrices(filename)
    filename = F'4_converged_wfn_b3lyp/c60_{step}-RESTART.wfn'
    _, _, _, mo_coeffs = CP2K_methods.read_wfn_file(filename)
    overlaps.append(data[0])
    coeffs.append(mo_coeffs)
    ks_mats.append(data[1])
    #clear_output()
overlaps = np.array(overlaps)
coeffs = np.array(coeffs)
ks_mats = np.array(ks_mats)
print(ks_mats.shape, coeffs.shape, overlaps.shape)
np.save('c60_b3lyp_converged_overlaps.npy', overlaps)
np.save('c60_b3lyp_converged_ks_mats.npy', ks_mats)
np.save('c60_b3lyp_converged_coeffs.npy', coeffs)

# Fourth the HSE06 converged wfns
istep = 999
fstep = 3001
overlaps = []
coeffs = []
ks_mats = []
for step in range(istep,fstep):
    print(step)
    filename = F'6_converged_wfn_hse06/c60_{step}-data-1_0.Log'
    data = CP2K_methods.read_ao_matrices(filename)
    filename = F'6_converged_wfn_hse06/c60_{step}-RESTART.wfn'
    _, _, _, mo_coeffs = CP2K_methods.read_wfn_file(filename)
    overlaps.append(data[0])
    coeffs.append(mo_coeffs)
    ks_mats.append(data[1])
    #clear_output()
overlaps = np.array(overlaps)
coeffs = np.array(coeffs)
ks_mats = np.array(ks_mats)
print(ks_mats.shape, coeffs.shape, overlaps.shape)
np.save('c60_hse06_converged_overlaps.npy', overlaps)
np.save('c60_hse06_converged_ks_mats.npy', ks_mats)
np.save('c60_hse06_converged_coeffs.npy', coeffs)

