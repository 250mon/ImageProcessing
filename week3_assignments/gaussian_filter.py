import numpy as np


def gaussian_2d(self, k_size=(3,3), sigma=1.0):
    """ Gaussian filter generation """
    # Make a row for width
    # (3,)
#    dist_w = (np.arange(k_size[1]) - (k_size[1]-1)/2)**2  # m**2
    dist_w = np.power((np.arange(k_size[1]) - (k_size[1]-1)/2), 2)
    # (1, 3)
#    dist_w = np.reshape(dist_w, (1,)+np.shape(dist_w)) # the same as dist_w.reshape(1, -1)
    dist_w = dist_w.reshape(1, -1)

    # Make a column for hight
    # (3,)
#    dist_h = (np.arange(k_size[0]) - (k_size[0]-1)/2)**2  # n**2
    dist_h = np.power((np.arange(k_size[0]) - (k_size[0]-1)/2), 2)
    # (3, 1)
#    dist_h = np.reshape(dist_h, np.shape(dist_h)+(1,)) # the same as dist_h.reshape(-1, 1)
    dist_h = dist_h.reshape(-1, 1)
    
    # Make a kernel size matrix for width
    w_dist = []
    for idx in range(k_size[0]):
        w_dist.append(dist_w)    # (3x1x3); dist_w:(1x3)
    w_dist = np.concatenate(w_dist, axis=0)    # (3x3)
    
    # Make a kernel size matrix for width
    h_dist = []    
    for idx in range(k_size[1]):
        h_dist.append(dist_h)    # (3x3x1); dist_h:(3x1)
    h_dist = np.concatenate(h_dist, axis=1)    # (3x3)

    # Make a Gaussian kernel
    ker = np.exp(-(h_dist + w_dist) / (2 * sigma ** 2))

    return ker/np.sum(ker)
