import numpy as np


def npy2csv(in_file, output):
    prediction = np.load(in_file).astype(int)
    with open(output, 'w') as f:
        print('Id,Response', file=f)
        for i in range(10000):
            print("%d,%d" % (i+20000, prediction[i]), file=f)
