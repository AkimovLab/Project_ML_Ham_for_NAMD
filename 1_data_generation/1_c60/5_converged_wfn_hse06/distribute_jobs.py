import os
import numpy as np
f = open('submit_template.slm', 'r')
lines = f.readlines()
f.close()

njobs = 61
init_step = 999
final_step = 3001
steps = list(np.linspace(init_step, final_step, njobs, dtype=int))
print(steps)

for j in range(len(steps)-1):
    istep = steps[j]
    fstep = steps[j+1]
    print(F'Running calculations for istep {istep} to fstep {fstep}')
    f = open('submit_1.slm','w')
    for i in range(len(lines)):
        if 'python' in lines[i]:
            f.write(F'python run_all.py --istep {istep} --fstep {fstep}\n')
        else:
            f.write(lines[i])
    f.close()
    os.system('sbatch submit_1.slm')


