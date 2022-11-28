import numpy as np
import statistic_model_pybind
import cvxpy as cp

def ising_generator(P, N, Edges, seed, strength=0.5):
    """
    Generate data for Ising model
    :param P: number of variables
    :param N: number of samples
    :param Edges: number of Edges
    :return: data (2^p * p+1), theta, coef
    """
    np.random.seed(seed)
    theta = np.zeros(shape=(P, P))
    E = np.random.choice(int(P*(P-1)/2), Edges, replace=False)
    flatten_theta = np.zeros(int(P*(P-1)/2))
    flatten_theta[E] = (np.random.randint(2, size=Edges) - 0.5) * 2 * strength

    idx = 0
    for i in range(P):
        for j in range(i+1,P):
            if idx in E:
                theta[i,j] = flatten_theta[idx]
                theta[j,i] = theta[i,j]
            idx += 1

    data = statistic_model_pybind.ising_generator(N, theta, seed)
    return data[np.where(data[:,0] > 0.5)[0],], theta, flatten_theta

def ising_loss(para, data):
    return statistic_model_pybind.ising_loss(para, np.zeros(0), data)

def ising_grad(para, data, compute_para_index):
    return statistic_model_pybind.ising_grad(para, np.zeros(0), data, compute_para_index)

class IsingData:
    def __init__(self, data):
        self.n = data.shape[0]
        self.p = data.shape[1] - 1
        self.table = data[:, 1:]
        self.freq = data[:, 0]

        self.index_translator = np.zeros(shape=(self.p, self.p), dtype=np.int32)
        idx = 0
        for i in range(self.p):
            for j in range(i+1, self.p):
                self.index_translator[i,j] = idx
                self.index_translator[j,i] = idx
                idx += 1

def ising_cvxpy(para, data):
    loss = 0.0
    for i in range(data.n):
        for k in range(data.p):
            tmp = 0.0
            for j in range(data.p):
                if j != k:
                    tmp -= 2 * para[data.index_translator[k,j]] * data.table[i,j] * data.table[i,k]
            loss += data.freq[i] * cp.logistic(tmp)
    return loss
