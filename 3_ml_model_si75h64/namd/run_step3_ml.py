import os
import time

sizes = [50, 100, 250, 500, 750, 1000]
functionals = ['b3lyp','hse06', 'pbe']
nthreads = 10

f = open('submit_1.slm', 'r') 
lines = f.readlines()
f.close()

c = 0
for size in sizes:
    for functional in functionals:
        c += 1
        f = open('submit_2.slm', 'w')
        for i in range(len(lines)):
            if 'python' in lines[i]:
                f.write(F'python step_3.py --functional {functional} --nthreads {nthreads} --size {size} --istep 0 --fstep 2000\n')
            else:
                f.write(lines[i])
        f.close()
        #time.sleep(3)
        os.system('sbatch submit_2.slm')
print(c) 

