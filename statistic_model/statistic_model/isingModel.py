import numpy as np
import jax.numpy as jnp
import statistic_model_pybind

def ising_generator(P, N, Edges, seed):
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
    flatten_theta[E] = np.random.randint(2, size=Edges) - 0.5

    idx = 0
    for i in range(P):
        for j in range(i+1,P):
            if idx in E:
                theta[i,j] = flatten_theta[idx]
                theta[j,i] = theta[i,j]
            idx += 1

    data = statistic_model_pybind.ising_generator(N, theta, seed)
    return data[np.where(data[:,0] > 0.5)[0],], theta, flatten_theta

class IsingData:
    """
    struct IsingData{
        MatrixXd table;
        VectorXd freq;
        const int p;
        IsingData(MatrixXd X): p(X.cols()-1) {
            std::vector<int> index;
            for(Eigen::Index i = 0; i < X.rows(); i++){
                if(X(i,0) > 0.5){
                    index.push_back(i);
                }
            }
            freq.resize(index.size());
            table.resize(index.size(),p);
            for(Eigen::Index i = 0; i < index.size(); i++){
                freq(i) = X(index[i],0);
                table.row(i) = X.row(index[i]).tail(p);
            }
        }
    };
    """
    def __init__(self, X):
        self.p = X.shape[1] - 1
        index = np.where(X[:,0] > 0.5)[0]
        self.freq = X[index,0]
        self.table = X[index,1:]


def ising_loss_no_intercept(para, data):
    """
    T loss = T(0.0);
    for(int i = 0; i < data->table.rows(); i++){
        int idx = 0;
        for(int s = 0; s < data->p; s++){
            for(int t = s+1; t < data->p; t++){
                loss -= 2 * data->freq(i) * data->table(i,s) * data->table(i,t) * para(idx++);
            }
        }
        for(int s = 0; s < data->p; s++){
            T tmp = T(0.0);
            for(int t = 0; t < data->p; t++){
                if(t > s)
                    tmp += para((2*data->p-s)*(s+1)/2+t-s-1-data->p) * data->table(i,t);
                else if(t < s)
                    tmp += para((2*data->p-t)*(t+1)/2+s-t-1-data->p) * data->table(i,t);
            }
            loss += data->freq(i) * log(1+exp(tmp));
        }
    }
    return loss;
    """
    loss = 0.0
    for i in range(data.table.shape[0]):
        idx = 0
        for s in range(data.p):
            for t in range(s+1,data.p):
                loss -= 2 * data.freq[i] * data.table[i,s] * data.table[i,t] * para[idx]
                idx += 1
        for s in range(data.p):
            tmp = 0.0
            for t in range(data.p):
                if t > s:
                    tmp += para[int((2*data.p-s)*(s+1)/2+t-s-1-data.p)] * data.table[i,t]
                elif t < s:
                    tmp += para[int((2*data.p-t)*(t+1)/2+s-t-1-data.p)] * data.table[i,t]
            loss += data.freq[i] * jnp.log(1+jnp.exp(tmp))
    return loss
