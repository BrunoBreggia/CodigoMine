import numpy as np

def numerical_estimator(X:np.ndarray, Y:np.ndarray, bins:float = 1000):
    assert X.shape == Y.shape, "Signals do not have same sizes"

    x_max, x_min = X.max(), X.min()
    y_max, y_min = Y.max(), Y.min()

    # bins = 1000
    x_step = (x_max - x_min) / bins
    y_step = (y_max - y_min) / bins

    x_bins = np.linspace(x_min, x_max, bins+1)
    y_bins = np.linspace(y_min, y_max, bins+1)

    x_indxs = np.digitize(X, x_bins) - 1
    y_indxs = np.digitize(Y, y_bins) - 1

    x_counter = np.zeros(shape=x_bins.size, dtype=np.float64)
    y_counter = np.zeros(shape=y_bins.size, dtype=np.float64)
    xy_counter = np.zeros(shape=(x_bins.size, y_bins.size), dtype=np.float64)

    x_counter[x_indxs] += 1
    y_counter[y_indxs] += 1
    xy_counter[x_indxs, y_indxs] += 1

    px = x_counter / X.size
    py = (y_counter / Y.size).T
    pxy = xy_counter / (X.size + Y.size)

    p1 = (pxy/px)/py
    p1 = np.nan_to_num(p1)
    p1[p1==0] = 1
    p2 = np.sum(pxy * np.log(p1))
    # MI = np.sum(pxy * np.log((pxy/px)/py))
    return p2
