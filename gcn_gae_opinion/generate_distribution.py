import numpy as np
import pickle
from collections import Counter


def get_omega(b, u):
    W = 2.0
    a = 0.5
    d = 1.0 - b - u
    r = W * b / u
    s = W * d / u
    alpha = r + W * a
    beta = s + W * (1.0 - a)
    return  np.round(alpha+beta)


def get_dc_dis_uncertain():
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/dc_belief_T38_0.8.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/dc_uncertain_T38_0.8.npy")
    dis = np.zeros([180, 39])
    for i in range(len(u_all)):
        ui = u_all[i]
        bi = b_all[i]
        obs = get_omega(bi, ui)
        cb = Counter(obs)
        for k in range(39):
            dis[i][k] = cb[k+2]
    np.save("/network/rit/lab/ceashpc/xujiang/traffic_data/DC_data/distribution_DC.npy", dis)
    return dis

def get_pa_dis_uncertain():
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_belief_T38_0.8.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_uncertain_T38_0.8.npy")
    dis = np.zeros([180, 39])
    for i in range(len(u_all)):
        ui = u_all[i]
        bi = b_all[i]
        obs = get_omega(bi, ui)
        cb = Counter(obs)
        for k in range(39):
            dis[i][k] = cb[k+2]
    np.save("/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/distribution_PA.npy", dis)
    return dis

if __name__ == '__main__':
    # get_dc_dis_uncertain()
    get_pa_dis_uncertain()
    dis = np.load("/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/distribution_PA.npy")
    dis_mean = np.mean(dis, axis=0)
    dis_std = np.std(dis, axis=0)
    print(1)