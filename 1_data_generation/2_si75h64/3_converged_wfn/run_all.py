import os
import numpy as np
import time
import libra_py.packages.cp2k.input as cp2k_input
import libra_py.packages.cp2k.methods as CP2K_methods


a_cell_size = 25.0
b_cell_size = 25.0
c_cell_size = 25.0
istep = 999
fstep = 3001
added_mos = 40
charge = 0
nprocs = 16
project = "silicon"
path_to_trajectory = "../1_md/Si75H64-300-pos-1.xyz"
path_to_submit_template = "submit_template.slm" 

f = open(path_to_submit_template, 'r') 
lines = f.readlines()
f.close()

h = {"element": "H", "basis_set":"ORB SZV-MOLOPT-GTH", "potential":"GTH-PBE-q1", "fit_basis_set":"cFIT3" }
c = {"element": "C", "basis_set":"ORB SZV-MOLOPT-GTH", "potential":"GTH-PBE-q4", "fit_basis_set":"cFIT3" }
o = {"element": "O", "basis_set":"ORB DZVP-MOLOPT-GTH", "potential":"GTH-PBE-q6", "fit_basis_set":"cFIT3" }
n = {"element": "N", "basis_set":"ORB DZVP-MOLOPT-GTH", "potential":"GTH-PBE-q5", "fit_basis_set":"cFIT3" }
ti = {"element": "Ti", "basis_set":"ORB DZVP-MOLOPT-SR-GTH", "potential":"GTH-PBE-q12", "fit_basis_set":"cFIT10" }
re = {"element": "Re", "basis_set":"ORB DZVP-MOLOPT-SR-GTH", "potential":"GTH-PBE-q15", "fit_basis_set":"cFIT10" }
fe = {"element": "Fe", "basis_set":"ORB DZVP-MOLOPT-SR-GTH", "potential":"GTH-PBE-q16", "fit_basis_set":"cFIT10" }
si = {"element": "Si", "basis_set":"ORB DZVP-MOLOPT-SR-GTH", "potential":"GTH-PBE-q4", "fit_basis_set":"cFIT3" }

dft_print = """
     &PRINT
       &AO_MATRICES
         &EACH
           QS_SCF 0
         &END
         !DENSITY T
         OVERLAP T
         KOHN_SHAM_MATRIX T
         FILENAME data
         NDIGITS 10
       &END
!      &MO
!        ENERGIES .TRUE.
!        COEFFICIENTS .TRUE.
!        OCCUPATION_NUMBERS .TRUE.
!        FILENAME coeffs
!        NDIGITS 8
!        &EACH
!          QS_SCF 0
!        &END
!     &END
    &END
"""


params = {"input_filename":"", "charge":charge, "multiplicity":1, "uks":".FALSE.",
         "wfn_restart_name": "test.wfn",
         "run_type": "ENERGY", "max_force": 7.0e-5, "max_disp": 0.002, "max_iter": 300,
         "project": "", "added_mos":added_mos, "smearing":False,
         "solver":"DIAG", "eps_scf": 1.0e-6, "max_scf": 60, "scf_guess": "ATOMIC",
         "poisson_solver": "MT", "eps_default": 1.0e-16,
         # XC for hybrid functionals
         "method":"manual_XC", 
         "functional_names": ["GGA_C_PBE", "GGA_X_PBE"], 
         #"functional_names": ["HYB_GGA_XC_HSE06"],
         #"functional_keys": [["SCALE", "_OMEGA_HF", "_OMEGA_PBE"]],"outer_scf": False,
         "functional_keys": [["SCALE"]],"outer_scf": False,
         "HF_exchange": False, "eps_schwarz": "1.0E-10", "interaction_potential": "SHORTRANGE",
         "admm_calculations": True, 
         "admm_purification_method": "NONE", #"MO_DIAG", 
         "cell.periodic":"NONE","center_coordinates":".TRUE.",
         "xyz_file": "coord.xyz", "kinds": [ h, c, o, ti, fe, re, n, si ], "dft_print": ""}

params.update({"dft_print": dft_print})
params.update({"cell.A":[a_cell_size, 0.0, 0.0]})
params.update({"cell.B":[0.0, b_cell_size, 0.0]})
params.update({"cell.C":[0.0, 0.0, c_cell_size]})

for step in range(istep, fstep):
    #time.sleep(0.8)
    print(F'Running calculations for step {step}...')
    CP2K_methods.read_trajectory_xyz_file(path_to_trajectory, step)
    params.update({"xyz_file": F"coord-{step}.xyz"})
    params.update({"project": F"{project}_{step}"})
    params.update({"input_filename":F"{project}_{step}.inp"})
    params.update({"functional_names": ["GGA_X_PBE", "GGA_C_PBE"]})
    params.update({"functional_keys":  [["SCALE"], ["SCALE"]]})
    params.update({"functional_key_vals":[[1.0], [1.0]]})
    params.update({"HF_exchange": False, "admm_calculations": False})
    params.update({"cell.A":[a_cell_size, 0.0, 0.0]})
    params.update({"cell.B":[0.0, b_cell_size, 0.0]})
    params.update({"cell.C":[0.0, 0.0, c_cell_size]})
    params.update({"eps_scf": 5.0e-7})
    cp2k_input.generate(params)


    f = open(F'submit_1.slm','w')
    f.write('#!/bin/bash\n')
    for i in range(len(lines)):
        if "SBATCH" in lines[i]:
            f.write(lines[i])
    f.write('\n')
    f.write('module load cp2k_v23\n')
    f.write('export OMP_NUM_THREADS=1 \n')
    f.write('export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so \n\n')
    f.write(F'mpirun -np {nprocs} cp2k.psmp -i {project}_{step}.inp -o {project}_{step}.log \n') 
    f.write(F'rm coord-{step}.xyz {project}_{step}.inp {project}_{step}-RESTART.wfn.bak-1 \n') 
    #f.write(F'rm {project}_{step}-RESTART.wfn.bak-1 \n') 
    #f.write(F'rm coord-{step}.xyz {project}_{step}.inp {project}_{step}.log \n')
    f.write(F'rm "slurm-$SLURM_JOBID.out" \n\n')
    f.close()
    os.system('sbatch submit_1.slm')
#    if step%1000==0 and step!=istep:
#        print('Waiting for other jobs to be done. Current step:', step)
#        time.sleep(60*18)



