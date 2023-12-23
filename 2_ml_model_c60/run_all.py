import os
import time

sizes = [50, 100, 250, 500, 750, 1000]
functionals = ['hse06', 'b3lyp', 'pbe']

f = open('submit_1.slm', 'r')
lines = f.readlines()
f.close()

for size in sizes:
    for functional in functionals:
        f = open('submit_2.slm', 'w')
        for i in range(len(lines)):
            if 'python' in lines[i]:
                f.write(F'python ml_model.py --size {size} --functional {functional} \n')
                #f.write(F'python compute_overlap_ml.py --size {size} --functional {functional} \n')
            else:
                f.write(lines[i])
        f.close()
        #time.sleep(3)
        os.system('sbatch submit_2.slm')
