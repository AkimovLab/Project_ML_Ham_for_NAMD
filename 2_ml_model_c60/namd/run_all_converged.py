import os
import time

functionals = ['hse06', 'pbe','b3lyp']
methods = ['fssh','ida','msdm'] 
istates = [16,26,37,47,57] 
ntraj = 250
nsteps = 2000
nthreads = 10

f = open('submit_1.slm', 'r') 
lines = f.readlines()
f.close()

c = 0
for functional in functionals:
    for method in methods:
        for istate in istates:
            c += 1
            f = open('submit_2.slm', 'w')
            for i in range(len(lines)):
                if 'python' in lines[i]:
                    f.write(F'python step_4_converged.py --functional {functional} --method {method} --nsteps {nsteps} --istate {istate} --ntraj {ntraj} --nthreads {nthreads} \n')
                else:
                    f.write(lines[i])
            f.close()
            os.system('sbatch submit_2.slm')
print(c)

