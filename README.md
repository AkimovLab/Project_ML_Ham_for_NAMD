# Project_ML_Ham_for_NAMD

This repository contains necessary files for running nonadiabatic molecular dynamics (NA-MD) using machine-leaning (ML) techniques.
The ML approach predicts the Hamiltonian matrix in one level of theory, PBE, B3LYP, and HSE06, from the Hamiltonian matrix computed from atomic-guess density using the PBE functional.
This approach is applied to two nanostructures of C60 fullerene and Si75H64 nanocluster with 240 and 1039 atomic orbital basis functions. Electronic structure calculations are done using CP2K software package.
ML is done using scikit-learn package. All other calculations and analysis, including pre- and post-pocessing data, computing molecular orbital overlaps in atomic orbital basis, reading and writing CP2K binary files, NA-MD with decoherence methods etc are done using Libra code.


`1_data_generation` contains CP2K inputs for computing the reference data for training and testing for both structures. It generates the Kohn-Sham (KS) Hamiltonian matrices for atomic-guess and for converged results using PBE, B3LYP, and HSE06 functionals. 
The data are then parsed using Libra code in `extract_data.py` and the matrices nd coefficients are stored in `.npy` files. 
The molecular dynamics trajectory for C60 is provided in the `trajectory_coordinates.tar.bz2` file and the trajectory for Si75H64 nanocluster, in `Si75H64-300-pos-1.zip` file, is adopted from our previous calculations in this link.


In `2_ml_model_c60` and `3_ml_model_si75h64` folders, the ML is performed by taking atomic-guess Hamiltonian matrices as input and converged Hamiltonian matrices from one level of theory mentioned above as outputs.
For ML, a kernel ridge regression (KRR) model is adopted and the models are trained and the predicted Hamiltonian matrices are produced. The energies and molecular orbitals coefficients are computed by solving the generalized KS eigenvalue problem and written in binary format that can be read by CP2K to compute the total energy. 


Using the molecular orbital coefficients from reference and ML predicted data, the overlap and time-overlap matrices are built to perform NA-MD. A mixed electron and hole single-particle excitation basis is made in `step_3_*.py` files.
NA-MD is performed to study hot-carrier relaxation dynamics for both structures using `step_4_*.py` files in `namd` folders. All analysis scripts related to this project are similar to the scripts already in Tutorials Libra repository of CompChemCyberTraining.


```
1_data_generation
├── 1_c60
│   ├── 1_md
│   │   ├── c60.xyz
│   │   ├── md.inp
│   │   ├── submit.slm
│   │   └── trajectory_coordinates.tar.bz2
│   ├── 2_atomic_wfn
│   │   ├── distribute_jobs.py
│   │   ├── run_all.py
│   │   └── submit_template.slm
│   ├── 3_converged_wfn
│   │   ├── distribute_jobs.py
│   │   ├── run_all.py
│   │   └── submit_template.slm
│   ├── 4_converged_wfn_b3lyp
│   │   ├── distribute_jobs.py
│   │   ├── run_all.py
│   │   └── submit_template.slm
│   ├── 5_converged_wfn_hse06
│   │   ├── distribute_jobs.py
│   │   ├── run_all.py
│   │   └── submit_template.slm
│   ├── 6_sample_molden_file
│   │   └── c60_2800-molden-1_0.molden
│   └── extract_data.py
└── 2_si75h64
    ├── 1_md
    │   └── Si75H64-300-pos-1.zip
    ├── 2_atomic_wfn
    │   ├── run_all.py
    │   └── submit_template.slm
    ├── 3_converged_wfn
    │   ├── run_all.py
    │   └── submit_template.slm
    ├── 4_converged_wfn_b3lyp
    │   ├── run_all.py
    │   └── submit_template.slm
    ├── 5_converged_wfn_hse06
    │   ├── run_all.py
    │   └── submit_template.slm
    ├── 6_sample_molden_file
    │   └── silicon_2800-molden-1_0.molden
    └── extract_data.py
2_ml_model_c60
├── compute_overlap_ml.py
├── ml_model.py
├── namd
│   ├── recipes
│   │   ├── dish_nbra.py
│   │   ├── fssh2_nbra.py
│   │   ├── fssh_nbra.py
│   │   ├── gfsh_nbra.py
│   │   ├── ida_nbra.py
│   │   ├── mash_nbra.py
│   │   └── msdm_nbra.py
│   ├── run_all_converged.py
│   ├── run_all_ml.py
│   ├── run_step3.py
│   ├── step_3_converged.py
│   ├── step_3_ml.py
│   ├── step_4_converged.py
│   ├── step_4_ml.py
│   └── submit_1.slm
├── run_all.py
├── shuffled_indices
│   └── shuffled_indices.npy
└── submit_1.slm
3_ml_model_si75h64
├── compute_overlap_ml.py
├── ml_model.py
├── namd
│   ├── recipes
│   │   ├── dish_nbra.py
│   │   ├── fssh2_nbra.py
│   │   ├── fssh_nbra.py
│   │   ├── gfsh_nbra.py
│   │   ├── ida_nbra.py
│   │   ├── mash_nbra.py
│   │   └── msdm_nbra.py
│   ├── run_all_converged.py
│   ├── run_all_ml.py
│   ├── run_step3_converged.py
│   ├── run_step3_ml.py
│   ├── step_3_converged.py
│   ├── step_3_ml.py
│   ├── step_4_converged.py
│   ├── step_4_ml.py
│   └── submit_1.slm
├── run_all.py
├── shuffled_indices
│   └── shuffled_indices.npy
└── submit_1.slm
```


