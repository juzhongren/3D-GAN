import numpy as np
def read_list(filename):
    c = np.loadtxt(filename)
    c = c.tolist()
    x = [int(x) for x in c]
    # print(x)
    return x

if __name__ == '__main__':
    read_list("test_1.txt")