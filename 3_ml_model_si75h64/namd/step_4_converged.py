import os, glob, time, h5py, warnings, sys

import multiprocessing as mp
import matplotlib.pyplot as plt   # plots
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit

from liblibra_core import *
import util.libutil as comn

import libra_py
from libra_py import units, data_conv #, dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
#import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
#import libra_py.data_savers as data_savers
import libra_py.workflows.nbra.decoherence_times as decoherence_times
import libra_py.data_visualize

from recipes import dish_nbra, fssh_nbra, fssh2_nbra, gfsh_nbra, ida_nbra, mash_nbra, msdm_nbra

import argparse

parser = argparse.ArgumentParser() 
#parser.add_argument('--size', type=int) 
parser.add_argument('--functional', type=str) 
parser.add_argument('--method', type=str) 
parser.add_argument('--nsteps', type=int) 
parser.add_argument('--ntraj', type=int) 
parser.add_argument('--nthreads', type=int) 
parser.add_argument('--istate', type=int) 
args = parser.parse_args() 

istep = 0
fstep = 1998
a = 50
NSTEPS = fstep - istep
#================== Read energies =====================
E = []
for step in range(istep,fstep):
    energy_filename = F"{args.functional}_converged/Hvib_sd_{step}_re.npz"
    energy_mat = sp.load_npz(energy_filename)[:,0:a][0:a,:]
    # For data conversion we need to turn np.ndarray to np.array so that 
    # we can use data_conv.nparray2CMATRIX
    E.append( energy_mat.todense().real ) 
E = np.array(E)
NSTATES = E[0].shape[0]
#================== Read time-overlap =====================
St = []
for step in range(istep,fstep):        
    St_filename = F"{args.functional}_converged/St_sd_{step}_re.npz"
    St_mat = sp.load_npz(St_filename)[:,0:a][0:a,:]
    St.append( np.array( St_mat.todense() ) )
St = np.array(St)
#================ Compute NACs and vibronic Hamiltonians along the trajectory ============    
NAC = []
Hvib = [] 
for c, step in enumerate(range(istep,fstep)):
    nac_filename = F"{args.functional}_converged/Hvib_sd_{step}_im.npz"
    nac_mat = sp.load_npz(nac_filename)[:,0:a][0:a,:]
    NAC.append( np.array( nac_mat.todense() ) )
    Hvib.append( E[c]*(1.0+1j*0.0)  - (0.0+1j)*nac_mat[:, :] )

NAC = np.array(NAC)
Hvib = np.array(Hvib)

print('Number of steps:', NSTEPS)
print('Number of states:', NSTATES)
print(E.shape, St.shape, NAC.shape, Hvib.shape)
#sys.exit(0)
#==== St = np.load(F'../converged_{args.functional}_time_overlap.npy')[:,106:120,106:120] 
#==== #St = np.load(F'../converged_{args.functional}_time_overlap.npy')[:,119:126,119:126] 
#==== E1 = -np.load(F'../{args.functional}_converged_energies.npy')[:,106:120] 
#==== #E1 = np.load(F'../{args.functional}_converged_energies.npy')[:,119:126] 
#==== p_s = 'hole'
#==== #p_s = 'electron'
#==== E = []
#==== for i in range(E1.shape[0]-1):
#====     E.append( np.diag(E1[i]) )
#==== E = np.array(E)
#==== print(np.diag(E1).shape)
#==== print(St.shape, E.shape)
#==== del E1
#==== NAC = []
#==== Hvib = []
#==== for i in range(St.shape[0]):
#====     nac = (St[i]-St[i].T)/(2*1.0*units.fs2au)
#====     NAC.append(nac)
#====     Hvib.append(E[i]*(1.0+1j*0.0) - (0.0+1j)*nac)
#==== NAC = np.array(NAC)
#==== Hvib = np.array(Hvib)
#==== print(NAC.shape, Hvib.shape)
#==== 
nsteps = St.shape[0]
NSTEPS = nsteps
print(F"Number of steps = {nsteps}")
nstates = St.shape[1]
NSTATES = nstates                                                            
print(F"Number of states = {nstates}")


class abstr_class:
    pass

def compute_model(q, params, full_id):
    timestep = params["timestep"]
    nst = params["nstates"]
    obj = abstr_class()

    obj.ham_adi = data_conv.nparray2CMATRIX( E[timestep] )
    obj.nac_adi = data_conv.nparray2CMATRIX( NAC[timestep] )
    obj.hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep] )
    obj.basis_transform = CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep] )
    
    return obj


# ================= Computing the energy gaps and decoherence times ===============
# Prepare the energies vs time arrays
HAM_RE = []
for step in range(E.shape[0]):
    HAM_RE.append( data_conv.nparray2CMATRIX( E[step] ) )

# Average decoherence times and rates
tau, rates = decoherence_times.decoherence_times_ave([HAM_RE], [0], NSTEPS, 0)

# Computes the energy gaps between all states for all steps
dE = decoherence_times.energy_gaps_ave([HAM_RE], [0], NSTEPS)

# Decoherence times in fs
deco_times = data_conv.MATRIX2nparray(tau) * units.au2fs

# Zero all the diagonal elements of the decoherence matrix
np.fill_diagonal(deco_times, 0)

# Saving the average decoherence times [fs]
np.savetxt('decoherence_times.txt',deco_times.real)

# Computing the average energy gaps
gaps = MATRIX(NSTATES, NSTATES)
for step in range(NSTEPS):
    gaps += dE[step]
gaps /= NSTEPS

rates.show_matrix("decoherence_rates.txt")
gaps.show_matrix("average_gaps.txt")
#sys.exit(0)

# Let's visualize the map of decoherence times:

# In[6]:


#%matplotlib notebook
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
avg_deco = np.loadtxt('decoherence_times.txt')
nstates = avg_deco.shape[0]
plt.imshow(np.flipud(avg_deco), cmap='hot', extent=(0,nstates,0,nstates))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('fs')
#plt.clim(vmin=0, vmax=30)
plt.title(F'Decoherence times')
plt.tight_layout()
plt.savefig('Decoherence_times.png')
plt.show()


nac = np.average(np.abs(NAC), axis=0).real*units.au2ev*1000
print(nac.shape)
plt.figure()
plt.imshow(np.flipud(nac), cmap='hot')#, extent=(0,NSTATES,0,NSTATES))#, vmin=0, vmax=100)
plt.xlabel('State index')
plt.ylabel('State index')
colorbar = plt.colorbar()
colorbar.ax.set_title('meV')
#plt.clim(vmin=0, vmax=40)
plt.title(F'Nonadiabatic couplings')
plt.tight_layout()
plt.savefig('Nonadiabatic_couplings.png', dpi=600)

# In[8]:


#================== Model parameters ====================
model_params = { "timestep":0, "icond":0,  "model0":0, "nstates":NSTATES }

#=============== Some automatic variables, related to the settings above ===================
#############
NSTEPS = args.nsteps
#############

dyn_general = { "nsteps":NSTEPS, "ntraj":args.ntraj, "nstates":NSTATES, "dt":0.5*units.fs2au,
                "decoherence_rates":rates, "ave_gaps":gaps,                
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),
                "mem_output_level":2,
                "properties_to_save":[ "timestep", "time","se_pop_adi", "sh_pop_adi" ],
                "prefix":F"NBRA", "prefix2":F"NBRA", "isNBRA":0, "nfiles": nsteps - 1
              }
##########################################################
#============== Select the method =====================
if args.method=='dish':
    dish_nbra.load(dyn_general); prf = "DISH"  # DISH
if args.method=='fssh':
    fssh_nbra.load(dyn_general); prf = "FSSH"  # FSSH
if args.method=='fssh2':
    fssh2_nbra.load(dyn_general); prf = "FSSH2"  # FSSH2
if args.method=='gfsh':
    gfsh_nbra.load(dyn_general); prf = "GFSH"  # GFSH
if args.method=='ida':
    ida_nbra.load(dyn_general); prf = "IDA"  # IDA
if args.method=='mash':
    mash_nbra.load(dyn_general); prf = "MASH"  # MASH
if args.method=='msdm':
    msdm_nbra.load(dyn_general); prf = "MSDM"  # MSDM
##########################################################

#=================== Initial conditions =======================
#============== Nuclear DOF: these parameters don't matter much in the NBRA calculations ===============
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1 }

#============== Electronic DOF: Amplitudes are sampled ========
elec_params = {"ndia":NSTATES, "nadi":NSTATES, "verbosity":-1, "init_dm_type":0}

###########
istate = args.istate
###########
elec_params.update( {"init_type":1,  "rep":1,  "istate":istate } )  # how to initialize: random phase, adiabatic representation

if prf=="MASH":
    istates = list(np.zeros(NSTATES))
    istates[istate] = 1.0
    elec_params.update( {"init_type":4,  "rep":1,  "istate":3, "istates":istates } )  # different initialization for MASH



def function1(icond):
    print('Running the calculations for icond:', icond)
    time.sleep( icond * 0.01 )
    rnd=Random()
    mdl = dict(model_params)
    mdl.update({"icond": icond})  #create separate copy
    dyn_gen = dict(dyn_general)
    dyn_gen.update({"icond": icond})  #create separate copy
    dyn_gen.update({"prefix":F"converged_{prf}_functional_{args.functional}_istate_{istate}_icond_{icond}", "prefix2":F"converged_{prf}_functional_{args.functional}_istate_{istate}_icond_{icond}" }) 
    res = tsh_dynamics.generic_recipe(dyn_gen, compute_model, mdl, elec_params, nucl_params, rnd)



################################
nthreads = args.nthreads
ICONDS = list(np.linspace(1,2000,nthreads*2, dtype=int, endpoint=False))
################################

pool = mp.Pool(nthreads)
pool.map(function1, ICONDS)
pool.close()                            
pool.join()


