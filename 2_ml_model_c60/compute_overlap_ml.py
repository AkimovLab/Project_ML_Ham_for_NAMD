import os
import sys
import time
import pickle 
import joblib
import numpy as np

import matplotlib.pyplot as plt
from liblibra_core import *
import libra_py.packages.cp2k.methods as CP2K_methods
from libra_py import units, molden_methods, data_conv
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--size', type=int) 
parser.add_argument('--functional', type=str) 
args = parser.parse_args() 

start_time = time.time() 

size = args.size #50
functional = args.functional
nprocs = 9



ml_eigenvectors = np.load(F'{functional}_ml_eigenvectors_size_{size}.npy')
converged_eigenvectors = np.load(F'{functional}_converged_eigenvectors.npy') 

# In[369]:
# In[378]:
sample_molden_file = '../1_data_generation/1_c60/6_sample_molden_file/c60_2800-molden-1_0.molden'
path_to_trajectory = '../1_data_generation/1_c60/1_md/C60-pos-1.xyz'

def compute_mo_overlaps(eigenvectors_set_1, eigenvectors_set_2, step_1, step_2, istep):
    """
    This is a local function specific to this notebook that will compute the molecular orbitals
    overlap and time-overlaps using the built-in functions in Libra. Since it is a local function,
    we can use the previously computed variables such as sample_molden_file, path_to_trajectory, etc.
    """
    is_spherical = True
    molden_file_1 = F'temp_{functional}_{step_1}_size_{size}.molden'
    molden_methods.write_molden_file(molden_file_1, sample_molden_file, path_to_trajectory, step_1)
    molden_file_2 = F'temp_{functional}_{step_2}_size_{size}.molden'
    molden_methods.write_molden_file(molden_file_2, sample_molden_file, path_to_trajectory, step_2)
    shell_1, l_vals = molden_methods.molden_file_to_libint_shell(molden_file_1, is_spherical)
    shell_2, l_vals = molden_methods.molden_file_to_libint_shell(molden_file_2, is_spherical)
    if molden_file_1==molden_file_2:
        os.system(F'rm  {molden_file_1}')
    else:
        os.system(F'rm  {molden_file_1}  {molden_file_2}')
    AO_S = compute_overlaps(shell_1,shell_2,nprocs)
    AO_S = data_conv.MATRIX2nparray(AO_S)
    new_indices = CP2K_methods.resort_molog_eigenvectors(l_vals)
    # eigenvectors_1 = converged_eigenvectors[0, :, new_indices]
    eigenvectors_1 = eigenvectors_set_1[step_1-istep, :, new_indices]
    eigenvectors_2 = eigenvectors_set_2[step_2-istep, :, new_indices]
    #print(eigenvectors_1.shape)
    MO_overlap = np.linalg.multi_dot([eigenvectors_1.T, AO_S, eigenvectors_2])
    #print(np.diag(MO_overlap))
    return MO_overlap[0:150,:][:,0:150]

istep = 999
#os.system(F'mkdir ml_b3lyp_overlap_size_{size} converged_b3lyp_overlap')
S_ml_all = []
S_converged_all = []
St_ml_all = []
St_converged_all = []
for step in range(999, 3000):
    t1 = time.time()
    print(step)
    S_ml_all.append(np.diag(compute_mo_overlaps(ml_eigenvectors, converged_eigenvectors, step, step, istep)))
    print(time.time()-t1)
    print(S_ml_all[step-istep].shape) 
    
S_ml_all = np.array(S_ml_all) 


np.save(F'{functional}_ml_and_converged_overlap_size_{size}.npy', S_ml_all)
print('Done with all calculations! Elapsed time:', time.time()-start_time)


