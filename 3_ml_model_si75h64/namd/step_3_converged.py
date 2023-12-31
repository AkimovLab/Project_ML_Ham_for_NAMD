import os
import numpy as np
import scipy.sparse as sp
from liblibra_core import *
from libra_py.workflows.nbra import step3
from libra_py import data_conv
import libra_py.packages.cp2k.methods as CP2K_methods
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--istep', type=int) 
parser.add_argument('--functional', type=str) 
parser.add_argument('--nthreads', type=int) 
parser.add_argument('--fstep', type=int) 
args = parser.parse_args() 
a = 182-10
b = 182+10
functional = args.functional
istep = args.istep
fstep = args.fstep
zero_mat = np.zeros((b-a, b-a))
#print(active_space)
#print(len(active_space))
S  = np.load(F'../converged_{args.functional}_overlap.npy')[:, a:b, a:b] 
St = np.load(F'../converged_{args.functional}_time_overlap.npy')[:, a:b, a:b] 
E  = np.load(F'../{args.functional}_converged_energies.npy')[:, a:b] 

os.system(F'mkdir {functional}_converged')
for i in range(istep, fstep):
    E_i = np.diag(E[i])
    St_i = St[i,:,:]
    S_i = S[i,:,:]
    print(E_i.shape, S_i.shape, St_i.shape, zero_mat.shape)
    E_i_block = data_conv.form_block_matrix(E_i,zero_mat,zero_mat,E_i)
    St_i_block = data_conv.form_block_matrix(St_i,zero_mat,zero_mat,St_i)
    S_i_block = data_conv.form_block_matrix(S_i,zero_mat,zero_mat,S_i)
    E_sparse = sp.csc_matrix(E_i_block)
    St_sparse = sp.csc_matrix(St_i_block)
    S_sparse = sp.csc_matrix(S_i_block)
    sp.save_npz(F'{functional}_converged/E_ks_{i}.npz', E_sparse)
    sp.save_npz(F'{functional}_converged/St_ks_{i}.npz', St_sparse)
    sp.save_npz(F'{functional}_converged/S_ks_{i}.npz', S_sparse)


params_mb_sd = {
          'lowest_orbital': a+1, 'highest_orbital': b, 'num_occ_states': 10, 'num_unocc_states': 10,
          'isUKS': 0, 'number_of_states': 10, 'tolerance': 0.01, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': args.nthreads,
          'is_many_body': False, 'time_step': 0.5, 'es_software': 'cp2k',
          'path_to_npz_files': F'{functional}_converged',
          'logfile_directory': '../../1_data_generation/2_si75h64/3_converged_wfn',
          'path_to_save_sd_Hvibs': F'{functional}_converged',
          'outdir': F'{functional}_converged', 'start_time': istep, 'finish_time': fstep-1, 'sorting_type': 'energy',
         }

step3.run_step3_sd_nacs_libint(params_mb_sd)


