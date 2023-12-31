import os
import sys
import time
import pickle 
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
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

nsplit = 200
size = args.size #50
functional = args.functional
#functional = 'b3lyp'
#functional = 'pbe'
nprocs = 9

# # 1. Loading data
# In this section, we load the Kohn-Sham and overlap matrices and molecular orbitals coefficients.

# In[327]:


atomic_ks_mats = np.load('../1_data_generation/2_si75h64/silicon_pbe_atomic_ks_mats.npy')#[0:1000,:,:]
converged_ks_mats = np.load(F'../1_data_generation/2_si75h64/silicon_{functional}_converged_ks_mats.npy')#[0:1000,:,:]
atomic_overlaps = np.load('../1_data_generation/2_si75h64/silicon_pbe_atomic_overlaps.npy')#[0:1000,:,:]
#converged_overlaps = np.load(F'../1_data_generation/2_si75h64/silicon_{functional}_converged_overlaps.npy')#[0:1000,:,:]
atomic_coeffs = np.load('../1_data_generation/2_si75h64/silicon_pbe_atomic_coeffs.npy')#[0:1000,:,:,:]
converged_coeffs = np.load(F'../1_data_generation/2_si75h64/silicon_{functional}_converged_coeffs.npy')#[0:1000,:,:,:]


# 1. First generate the upper indices of the KS matrix
upper_indices = np.triu_indices(atomic_ks_mats[0].shape[0])
atomic_ks_mats_1 = []
converged_ks_mats_1 = []
# 2. Extract the upper matrixas a vector
for i in range(atomic_ks_mats.shape[0]):
    upper_vector = atomic_ks_mats[i][upper_indices]
    atomic_ks_mats_1.append(upper_vector)
    upper_vector = converged_ks_mats[i][upper_indices]
    converged_ks_mats_1.append(upper_vector)
atomic_ks_mats_1 = np.array(atomic_ks_mats_1)
converged_ks_mats_1 = np.array(converged_ks_mats_1)
print(atomic_ks_mats_1.shape, converged_ks_mats_1.shape)


# # 2. Preparing inputs and outputs + Normalization
# 
# Let's try this using multiple models for which we train using the data split into multiple sets. The reason is that the dimensions are way to high specially for large systems. By `nsplit`, we split the data into multiple sets which are almost equal in size except in case the dimension is not divisible by split which will be different by `1`. 
# Then, for each set we use a regression based model. We can make the model for each atomic orbital (each basis function) but that is might increase the computational cost for a large matrix. For now let's choose this method.

# In[333]:


split_points = np.linspace(0,converged_ks_mats_1.shape[1], nsplit, dtype=int, endpoint=True)
print(split_points)


# ## Creating the list of indices for training the models
# 
# In this part, we select the indices randomly. It is better to proceed with the most distinct geometries but I'll put it in the **TODO** list.

# In[379]:


shuffled_indices = np.load('shuffled_indices/shuffled_indices.npy')
print(shuffled_indices)


# In[53]:


# shuffled_indices = np.arange(atomic_ks_mats.shape[0])
# np.random.shuffle(shuffled_indices)
# print(shuffled_indices, shuffled_indices.shape)
# np.save('shuffled_indices_b3lyp.npy', shuffled_indices)


# In[349]:


# The size of the training set
train_indices = shuffled_indices[0:size]
test_indices = shuffled_indices[size:]
# ==========
overlap_train = atomic_overlaps[train_indices,:]
overlap_test = atomic_overlaps[test_indices,:]
coeffs_conv_train = converged_coeffs[train_indices]
coeffs_conv_test = converged_coeffs[test_indices]
coeffs_atomic_train = atomic_coeffs[train_indices]
coeffs_atomic_test = atomic_coeffs[test_indices]

inputs_train = []
inputs_test = []
outputs_train = []
outputs_test = []

for i in range(len(split_points)-1):
    start = split_points[i]
    end = split_points[i+1]
    inputs_train.append(atomic_ks_mats_1[train_indices,start:end])
    outputs_train.append(converged_ks_mats_1[train_indices,start:end])
    inputs_test.append(atomic_ks_mats_1[test_indices,start:end])
    outputs_test.append(converged_ks_mats_1[test_indices,start:end])
print(len(inputs_train))
print(len(inputs_train[0]))
print(len(inputs_train[0][0]))


# In[350]:


# np.save('train_indices_b3lyp.npy', train_indices)
# np.save('test_indices_b3lyp.npy', test_indices)


# In[351]:


input_scalers = []
output_scalers = []
inputs_train_scaled = []
inputs_test_scaled = []
outputs_train_scaled = []
outputs_test_scaled = []
for i in range(nsplit-1):
    #print(i)
    #print(np.array(inputs_train).shape)
    input_scaler = StandardScaler()     # MinMaxScaler()
    output_scaler = StandardScaler()    # MinMaxScaler()
    input_scaler.fit(inputs_train[i])
    output_scaler.fit(outputs_train[i])
    
    # Scaling the training set
    inputs_train_scaled.append(input_scaler.transform(inputs_train[i]))
    outputs_train_scaled.append(output_scaler.transform(outputs_train[i]))
    # Scaling the testing set
    inputs_test_scaled.append(input_scaler.transform(inputs_test[i]))
    outputs_test_scaled.append(output_scaler.transform(outputs_test[i]))
    input_scalers.append(input_scaler)
    output_scalers.append(output_scaler)


# # 3. Making and training the model

# In[352]:


try:
    del models
except:
    pass
models = []

model_train_rmse = []
model_test_rmse = []
data_train_rmse = []
data_test_rmse = []

model_train_maes = []
model_test_maes = []
data_train_maes = []
data_test_maes = []

t1 = time.time()
for i in range(nsplit-1):
    print(inputs_test[i].shape)
    print(outputs_test[i].shape)
    # model = KernelRidge(kernel='poly', degree=1)
    model = KernelRidge(kernel='linear')
    model.fit(inputs_train_scaled[i], outputs_train_scaled[i])
    predictions_train = model.predict(inputs_train_scaled[i])  
    predictions_train_scaled = output_scalers[i].inverse_transform(predictions_train)
    # Mean square error
    mse = np.sqrt(mean_squared_error(outputs_train[i], predictions_train_scaled))
    model_train_rmse.append(mse)
    # Mean absolute error
    acc = mean_absolute_error(outputs_train[i], predictions_train_scaled)
    model_train_maes.append(acc)
    print(F'Model {i} ---> Training    MSE:', mse, '  MAE:', acc)
    predictions_test = model.predict(inputs_test_scaled[i])
    predictions_test_scaled = output_scalers[i].inverse_transform(predictions_test)
    # Mean square error
    mse = np.sqrt(mean_squared_error(outputs_test[i], predictions_test_scaled))
    model_test_rmse.append(mse)
    # Mean absolute error
    acc = mean_absolute_error(outputs_test[i], predictions_test_scaled)
    model_test_maes.append(acc)
    print(F'Model {i} ---> Testing     MSE:', mse, '  MAE:', acc)
    acc = mean_absolute_error(inputs_train[i], outputs_train[i])
    mse = np.sqrt(mean_squared_error(inputs_train[i], outputs_train[i]))
    data_train_maes.append(acc)
    data_train_rmse.append(mse)
    print(F'Split {i}: KS matrix MAE of atomic_wfn and converged_wfn for the train data:', acc)
    acc = mean_absolute_error(inputs_test[i], outputs_test[i])
    mse = mean_absolute_error(inputs_test[i], outputs_test[i])
    data_test_maes.append(acc)
    data_test_rmse.append(mse)
    print(F'Split {i}: KS matrix MAE of atomic_wfn and converged_wfn for the test data: ', mean_absolute_error(inputs_test[i], outputs_test[i]))
    print('====================================================')
    models.append(model)


# In[353]:
np.save(F'train_time_{functional}_model_size_{size}.npy', np.array( [time.time()-t1] ) ) 

np.save(F'{functional}_model_train_maes_size_{size}.npy', model_train_maes)
np.save(F'{functional}_model_test_maes_size_{size}.npy', model_test_maes)
np.save(F'{functional}_data_train_maes_size_{size}.npy', data_train_maes)
np.save(F'{functional}_data_test_mae_size_{size}.npy', data_test_maes)

np.save(F'{functional}_model_train_rmse_size_{size}.npy', model_train_rmse)
np.save(F'{functional}_model_test_rmse_size_{size}.npy', model_test_rmse)
np.save(F'{functional}_data_train_rmse_size_{size}.npy', data_train_rmse)
np.save(F'{functional}_data_test_rmse_size_{size}.npy', data_test_rmse)

#sys.exit(0)

os.system(F'mkdir {functional}_models')
for i in range(len(models)):
    joblib.dump(models[i], F'{functional}_models/model_{i}_size_{size}.pkl')



# # 4. Using the model
# ## Solve the Kohn-Sham equations
# 
# $$Kc=Sc\epsilon$$
# 
# 
# $$S=U^TU$$
# 
# 
# $$Kc=U^T Uc\epsilon$$
# 
# 
# $$(U^T)^{-1}KU^{-1}Uc=Uc\epsilon$$
# 
# 
# $$K'c'=c'\epsilon$$
# 
# 
# $$K'=(U^T)^{-1}KU^{-1}, Uc=c'$$
# 
# 
# $$c=U^{-1}c'$$
# 
# The density matrix is computed as follows:
# 
# $$P=2\times c_{occ}\times c_{occ}^T$$
# 
# Then the convergence error is computed from the following commutation relation:
# 
# $$e=KPS-SPK$$

# ## Using the model for test set

# In[357]:
# Defining an auxiliary function
def upper_vector_to_symmetric_nparray(upper_vector, upper_indices, mat_shape):
    """
    This function gets the upper triangular part of a matrix as a vector and retuns a symmetric matrix
    Args:
        upper_vector (nparray): The upper triangular of a matrix
        upper_indices (nparray): The indices of the upper part of the matrix
        mat_shape (tuple): The shape of the original numpy array
    Returns:
        matrix (nparray): The symmetric matix built based on the upper triangular matrix
    """
    matrix = np.zeros(mat_shape)
    matrix[upper_indices] = upper_vector
    matrix = matrix + matrix.T - np.diag(matrix.diagonal())
    return matrix


# Some reference for reading and writing CP2K wfn files
filename = '../1_data_generation/2_si75h64/3_converged_wfn/silicon_1000-RESTART.wfn'
basis_data, spin_data, eigen_vals_and_occ_nums, mo_coeffs = CP2K_methods.read_wfn_file(filename)
#CP2K_methods.write_wfn_file(output_name, basis_data, spin_data, eigen_vals_and_occ_nums, [eigenvectors])

timings = []
production_timing = []
atomic_error = []
ml_error = []
istep = 999 # The initial geometry from the MD trajectories
ml_energies= []
converged_energies = []
ml_eigenvectors = []
converged_eigenvectors = []
atomic_eigenvalues = []
atomic_eigenvectors = []
os.system(F'mkdir ml_{functional}_wfns')
for step in range(len(converged_ks_mats_1)):
#for step in range(1600,2002):#len(converged_ks_mats_1)):
    ks_mat = []
    t1 = time.time()
    for i in range(nsplit-1):
        start = split_points[i]
        end = split_points[i+1]
        input_test_scaled = input_scalers[i].transform(atomic_ks_mats_1[step,start:end][np.newaxis,:])
        #input_test_scaled = input_scalers[i].transform(converged_ks_mats_1[step,start:end][np.newaxis,:])
        #input_test_scaled = input_scalers[i].transform(inputs_test[i][step][np.newaxis,:])
        predict = models[i].predict(input_test_scaled)
        predict_scaled = output_scalers[i].inverse_transform(predict)
        ks_mat.append(predict_scaled)
    ks_mat = np.concatenate(ks_mat, axis=1)
    ks_mat = upper_vector_to_symmetric_nparray(ks_mat, upper_indices, converged_ks_mats[0].shape)
    production_timing.append(time.time()-t1)
    t1 = time.time()
    #overlap = atomic_overlaps[test_indices[step]]
    overlap = atomic_overlaps[step]
    eigenvalues, eigenvectors = CP2K_methods.compute_energies_coeffs(ks_mat, overlap)
    ml_energies.append(eigenvalues)
    ml_eigenvectors.append(eigenvectors)
    timings.append(time.time()-t1) 
    print(F'================= Step {step} ================ Elapsed time:', time.time()-t1)
    ks_mat_converged = converged_ks_mats[step]
    #ks_mat_converged = atomic_ks_mats[step]
    eigenvalues_converged, eigenvectors_converged = CP2K_methods.compute_energies_coeffs(ks_mat_converged, overlap)
    converged_energies.append(eigenvalues_converged)
    converged_eigenvectors.append(eigenvectors_converged)
    
    ks_mat_atomic = atomic_ks_mats[step]
    #ks_mat_converged = atomic_ks_mats[step]
    eigenvalues_atomic, eigenvectors_atomic = CP2K_methods.compute_energies_coeffs(ks_mat_atomic, overlap)
    atomic_eigenvalues.append(eigenvalues_atomic)
    atomic_eigenvectors.append(eigenvectors_atomic)
    
    #conv_coeff = converged_coeffs[test_indices[step]]
    #atomic_coeff = atomic_coeffs[test_indices[step]]
    conv_coeff = converged_coeffs[step]
    atomic_coeff = atomic_coeffs[step]
    #print(conv_coeff[0])
    #print(eigenvectors)
    print('Converged HOMO-LUMO gap:', (eigenvalues_converged[182]-eigenvalues_converged[181])*units.au2ev)
    print('ML HOMO-LUMO gap:', (eigenvalues[182]-eigenvalues[181])*units.au2ev)
    print('Atomic guess (PBE guess) HOMO-LUMO gap:', (eigenvalues_atomic[182]-eigenvalues_atomic[181])*units.au2ev)
    #if step==shuffled_indices[-1]:
    #output_name = F'ml_{functional}_wfns/ml_silicon_{functional}_size_{size}_{step+istep}-RESTART.wfn' 
    #CP2K_methods.write_wfn_file(output_name, basis_data, spin_data, eigen_vals_and_occ_nums, [eigenvectors[0:222,:]])
    error_ml = np.average(np.abs(np.abs(eigenvectors)[0:222,:] - np.abs(conv_coeff[0])))
    ml_error.append(error_ml)
    error_atomic = np.average( np.abs(np.abs(atomic_coeff[0]) - np.abs(conv_coeff[0] )))
    atomic_error.append(error_atomic)
    print('Atomic wfn error:', error_atomic)
    print('ML wfn error:    ', error_ml)

del atomic_ks_mats_1
del atomic_ks_mats
del atomic_overlaps
# In[368]:

timings = np.array(timings)
np.save(F'timings_{functional}_size_{size}.npy', timings)
production_timing = np.array(production_timing)
np.save(F'production_timings_{functional}_size_{size}.npy', production_timing)

#sys.exit(0)

ml_eigenvectors = np.array(ml_eigenvectors)
np.save(F'{functional}_ml_eigenvectors_size_{size}.npy', ml_eigenvectors)
ml_energies = np.array(ml_energies)
np.save(F'{functional}_ml_energies_size_{size}.npy', ml_energies)

converged_eigenvectors = np.array(converged_eigenvectors)
if not os.path.exists(F'{functional}_converged_eigenvectors.npy'):
    np.save(F'{functional}_converged_eigenvectors.npy', converged_eigenvectors)
converged_energies = np.array(converged_energies)
if not os.path.exists(F'{functional}_converged_energies.npy'):
    np.save(F'{functional}_converged_energies.npy', converged_energies)

ml_error = np.array(ml_error)
np.save(F'{functional}_ml_error_size_{size}.npy', ml_error)
atomic_error = np.array(atomic_error)
if not os.path.exists(F'{functional}_atomic_error.npy'):
    np.save(F'{functional}_atomic_error.npy', atomic_error)

atomic_eigenvectors = np.array(atomic_eigenvectors)
if not os.path.exists(F'{functional}_atomic_eigenvectors.npy'):
    np.save(F'{functional}_atomic_eigenvectors.npy', atomic_eigenvectors)
atomic_eigenvalues = np.array(atomic_eigenvalues)
if not os.path.exists(F'{functional}_atomic_eigenvalues.npy'):
    np.save(F'{functional}_atomic_eigenvalues.npy', atomic_eigenvalues)


# In[369]:

print('For functional:', functional, ' and size: ', size)
print('ML energies average error:', np.average(np.abs(ml_energies-converged_energies))*units.au2ev)
print('PBE initial guess energies average error:', np.average(np.abs(atomic_eigenvalues-converged_energies))*units.au2ev)

# Compute overlaps and time-overlaps separately!


# ## Creating molden files and computing overlaps and time-overlaps
# 
# In this section, we make the molden files for a geometry based on a sample molden file that contains the basis set data. 

# In[378]:


sample_molden_file = '../1_data_generation/2_si75h64/5_sample_molden_file/silicon_2800-molden-1_0.molden'
path_to_trajectory = '../1_data_generation/2_si75h64/1_md/Si75H64-300-pos-1.xyz'

def compute_mo_overlaps(eigenvectors_set, step_1, step_2, istep):
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
    eigenvectors_1 = eigenvectors_set[step_1-istep, :, new_indices]
    eigenvectors_2 = eigenvectors_set[step_2-istep, :, new_indices]
    #print(eigenvectors_1.shape)
    MO_overlap = np.linalg.multi_dot([eigenvectors_1.T, AO_S, eigenvectors_2])
    #print(np.diag(MO_overlap))
    return MO_overlap

istep = 999
#os.system(F'mkdir ml_b3lyp_overlap_size_{size} converged_b3lyp_overlap')
S_ml_all = []
S_converged_all = []
St_ml_all = []
St_converged_all = []
for step in range(999,3000):
    print(step)
    S_ml_all.append(compute_mo_overlaps(ml_eigenvectors, step, step, istep))
    S_converged_all.append(compute_mo_overlaps(converged_eigenvectors, step, step, istep))
    St_ml_all.append(compute_mo_overlaps(ml_eigenvectors, step, step+1, istep))
    St_converged_all.append(compute_mo_overlaps(converged_eigenvectors, step, step+1, istep))
    
S_ml_all = np.array(S_ml_all)
St_ml_all = np.array(St_ml_all)
S_converged_all = np.array(S_converged_all)
St_converged_all = np.array(St_converged_all)

np.save(F'ml_{functional}_overlap_size_{size}.npy', S_ml_all)
if not os.path.exists(F'converged_{functional}_overlap.npy'):
    np.save(F'converged_{functional}_overlap.npy',      S_converged_all)
np.save(F'ml_{functional}_time_overlap_size_{size}.npy', St_ml_all)
if not os.path.exists(F'converged_{functional}_time_overlap.npy'):
    np.save(F'converged_{functional}_time_overlap.npy',      St_converged_all)


print('Done with all calculations! Elapsed time:', time.time()-start_time)


