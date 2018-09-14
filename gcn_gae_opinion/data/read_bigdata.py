import numpy as np
import pickle
from collections import Counter
from scipy import sparse

hours = [6, 9, 12, 15, 18, 21]


def read_bigdata_re():
    feat_bigdata = []
    ref_ratios = [0.8]
    datasets = ['dc', 'philly']
    for ref_ratio in ref_ratios[:1]:
        for dataset in datasets[0:1]:
            for weekday in range(5)[1:2]:
                for hour in hours:
                    pkl_file = open(
                        '/network/rit/lab/ceashpc/xujiang/GAT/GAT1/data/traffic_data_directed/raw_network_{}_weekday_{}_hour_{}_refspeed_{}.pkl'.format(
                            dataset, weekday, hour, ref_ratio), 'rb')
                    feat_hour = []
                    [V, E, Obs] = pickle.load(pkl_file)
                    for edge in E:
                        feat_hour.append(Obs[edge])
                    feat_bigdata.append(feat_hour)
    return feat_bigdata


def find_neigh_edge(E, i):
    neigh = []
    nodes = E[i]
    for j in range(len(E)):
        if j != i:
            for node in nodes:
                if node in E[j]:
                    neigh.append(j)
                    break
    return neigh


def find_direct_neigh_edge(E, i):
    neigh = []
    nodes = E[i]
    for j in range(len(E)):
        if j != i:
            if nodes[1] == E[j][0]:
                neigh.append(j)
            elif nodes[0] == E[j][1]:
                neigh.append(j)
    return neigh


def get_ad_matrix():
    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/Beijing_Taxi/20130915ref_proc-0.9.pkl")
    [V, E, Obs] = pickle.load(pkl_file)
    ad_m = np.zeros([len(E), len(E)])
    for i in range(len(E)):
        neigh_i = find_neigh_edge(E, i)
        for k in neigh_i:
            k = int(k)
            ad_m[i][k] = int(1)

    return ad_m


def get_ad_matrix_direct():
    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-500-T-38-rate-0.1-testratio-0.2-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    ad_m = np.zeros([len(E), len(E)])
    for i in range(len(E)):
        neigh_i = find_direct_neigh_edge(E, i)
        if neigh_i == []:
            print(2)
        for k in neigh_i:
            k = int(k)
            ad_m[i][k] = int(1)

    return ad_m


def trim(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


def beta_to_opinion(omega, W=2.0, a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    alpha = omega[1]
    beta = omega[0]
    b = trim((alpha - W * a) / float(alpha + beta))
    d = trim((beta - W * (1 - a)) / float(alpha + beta))
    u = trim(W / float(alpha + beta + W))
    return [b, d, u, a]


def get_alphabeta(ome, W=2.0, a=0.5):
    r = ome[3]
    s = ome[4]
    alpha = r + W * a
    beta = s + W * (1 - a)
    return [alpha, beta]


def get_omega_obs(obs):
    W = 2.0
    r = Counter(obs)[1]
    s = Counter(obs)[0]
    u = W / (W + r + s)
    b = r / (W + r + s)
    d = s / (W + r + s)
    return [b, d, u, r, s]


def get_neigh_E(E):
    id_2_edge = {}
    edge_2_id = {}
    for idx, e in enumerate(E):
        id_2_edge[idx] = e
        edge_2_id[e] = idx
    edges_all = {}
    for e in E:
        e_id = edge_2_id[e]
        start_v = e[0]
        end_v = e[1]
        if edges_all.has_key(end_v):
            edges_all[end_v][e_id] = 1
        else:
            edges_all[end_v] = {e_id: 1}
        if edges_all.has_key(start_v):
            edges_all[start_v][e_id] = 1
        else:
            edges_all[start_v] = {e_id: 1}
    edge_up_nns = {}
    edge_down_nns = {}
    neigh = []
    for e in E:
        start_v = e[0]
        end_v = e[1]
        e_id = edge_2_id[e]
        if edges_all.has_key(start_v):
            edge_up_nns[e_id] = edges_all[start_v]
        if edges_all.has_key(end_v):
            edge_down_nns[e_id] = edges_all[end_v]
        neigh_i = np.hstack((edges_all[start_v].keys(), edges_all[end_v].keys()))
        neigh.append(neigh_i)
    return neigh


def get_bn_undirect_beijing():
    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/Beijing_Taxi/20130915ref_proc-0.9.pkl")
    [V, Ed, Obs] = pickle.load(pkl_file)
    E = Ed.keys()
    # E = Ed
    # E_ori = [e for e in E]
    print("load data")
    # neigh = get_neigh_E(E)
    omega_rs = {}
    opinion = {}
    edge_double = []
    for edge in E:
        edge_ = (edge[1], edge[0])
        if edge_ in E:
            E.remove(edge_)
            edge_double.append(edge)
    print("remove double edge")
    for edge in E:
        omega_rs[edge] = get_omega_obs(Obs[edge])
        opinion[edge] = get_alphabeta(omega_rs[edge])
    print("get opinion")
    for edge_d in edge_double:
        edge_d_ = (edge_d[1], edge_d[0])
        ome1 = get_omega_obs(Obs[edge_d])
        ome2 = get_omega_obs(Obs[edge_d_])
        op1 = beta_to_opinion(ome1)
        op2 = beta_to_opinion(ome2)
        omega_rs[edge_d] = np.mean(np.vstack((ome1, ome2)), axis=0)
        opinion[edge_d] = np.mean(np.vstack((op1, op2)), axis=0)

        # Obs[edge_d] = map(add, Obs[edge_d], Obs[edge_d_])
    print("average double edge")
    omega_f = []
    opinoin_f = []
    belief = np.zeros(len(E))
    uncertain = np.zeros(len(E))
    for i in range(len(E)):
        omega_f.append(omega_rs[E[i]])
        opinoin_f.append(opinion[E[i]])
        belief[i] = omega_rs[E[i]][0]
        uncertain[i] = omega_rs[E[i]][2]
    return omega_f, opinoin_f, belief, uncertain, opinion


def get_ad_matrix_beijing():
    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/Beijing_Taxi/20130915ref_proc-0.9.pkl")
    [V, Ed, Obs] = pickle.load(pkl_file)
    E = Ed.keys()
    # E = Ed
    E_ori = [e for e in E]
    print("load data")
    neigh = get_neigh_E(E)
    np.save("20130915ref_proc-0.9_neigh.npy", neigh)
    omega = {}
    opinion = {}
    edge_double = []
    for edge in E:
        edge_ = (edge[1], edge[0])
        if edge_ in E:
            E.remove(edge_)
            edge_double.append(edge)

    ad_m = np.zeros([len(E), len(E)])
    for i in range(len(E)):
        neigh_i = neigh[i]
        if neigh_i == []:
            print("none nieghbor")
        for k in neigh_i:
            k = int(k)
            ad_m[i][k] = int(1)
        print("process", i)
    ad_m = ad_m - np.eye(len(E))
    # np.save("adj_undirect_beijing.npy", ad_m)
    return ad_m


# def get_dc_data():
#     pkl_file = open(
#         "/network/rit/lab/ceashpc/xujiang/traffic_data/DC_data/tmp/raw_network_dc_weekday_0_hour_6_refspeed_0.9.pkl")
#     [V, E, Obs] = pickle.load(pkl_file)
#     omega_rs = {}
#     opinion = {}
#     for edge in E:
#         omega_rs[edge] = get_omega_obs(Obs[edge])
#         opinion[edge] = get_alphabeta(omega_rs[edge])
#     belief = np.zeros(len(E))
#     uncertain = np.zeros(len(E))
#     for i in range(len(E)):
#         belief[i] = omega_rs[E[i]][0]
#         uncertain[i] = omega_rs[E[i]][2]
#     return belief, uncertain


def get_bj_data():
    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/data/Beijing_Taxi/20130915ref_proc-0.9.pkl")
    [V, Ed, Obs] = pickle.load(pkl_file)
    E = Ed.keys()
    omega_rs = {}
    opinion = {}
    for edge in E:
        omega_rs[edge] = get_omega_obs(Obs[edge])
        opinion[edge] = get_alphabeta(omega_rs[edge])
    belief = np.zeros(len(E))
    uncertain = np.zeros(len(E))
    for i in range(len(E)):
        belief[i] = omega_rs[E[i]][0]
        uncertain[i] = omega_rs[E[i]][2]
    return belief, uncertain


def get_pa_data():
    feat_belief = []
    feat_uncertain = []
    ref_ratios = [0.8, 0.9]
    datasets = ['dc', 'philly']
    for ref_ratio in ref_ratios[1:2]:
        for dataset in datasets[1:2]:
            for weekday in range(5):
                for hour in hours:
                    pkl_file = open(
                        '/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/tmp/raw_network_{}_weekday_{}_hour_{}_refspeed_{}.pkl'.format(
                            dataset, weekday, hour, ref_ratio), 'rb')
                    [V, E, Obs] = pickle.load(pkl_file)
                    omega_rs = {}
                    opinion = {}
                    for edge in E:
                        omega_rs[edge] = get_omega_obs(Obs[edge])
                        opinion[edge] = get_alphabeta(omega_rs[edge])
                    belief = np.zeros(len(E))
                    uncertain = np.zeros(len(E))
                    for i in range(len(E)):
                        belief[i] = omega_rs[E[i]][0]
                        uncertain[i] = omega_rs[E[i]][2]
                    feat_belief.append(belief)
                    feat_uncertain.append(uncertain)
    return feat_belief, feat_uncertain


def get_dc_data():
    feat_belief = []
    feat_uncertain = []
    ref_ratios = [0.8, 0.9]
    datasets = ['dc', 'philly']
    for ref_ratio in ref_ratios[1:2]:
        for dataset in datasets[0:1]:
            for weekday in range(5):
                for hour in hours:
                    pkl_file = open(
                        '/network/rit/lab/ceashpc/xujiang/traffic_data/DC_data/tmp/raw_network_{}_weekday_{}_hour_{}_refspeed_{}.pkl'.format(
                            dataset, weekday, hour, ref_ratio), 'rb')
                    [V, E, Obs] = pickle.load(pkl_file)
                    omega_rs = {}
                    opinion = {}
                    for edge in E:
                        omega_rs[edge] = get_omega_obs(Obs[edge])
                        opinion[edge] = get_alphabeta(omega_rs[edge])
                    belief = np.zeros(len(E))
                    uncertain = np.zeros(len(E))
                    for i in range(len(E)):
                        belief[i] = omega_rs[E[i]][0]
                        uncertain[i] = omega_rs[E[i]][2]
                    feat_belief.append(belief)
                    feat_uncertain.append(uncertain)
    return feat_belief, feat_uncertain


def get_dc_data_window():
    window_size = 38
    feat_belief = []
    feat_uncertain = []
    ref_ratios = [0.8, 0.9]
    datasets = ['dc', 'philly']
    for ref_ratio in ref_ratios[0:1]:
        for dataset in datasets[1:2]:
            for weekday in range(5):
                for hour in hours:
                    pkl_file = open(
                        '/network/rit/lab/ceashpc/xujiang/traffic_data/PA_data/tmp2/raw_network_{}_weekday_{}_hour_{}_refspeed_{}.pkl'.format(
                            dataset, weekday, hour, ref_ratio), 'rb')
                    [V, E, Obs] = pickle.load(pkl_file)
                    for i in range(6):
                        t_Obs = {e: e_Obs[i:i + window_size] for e, e_Obs in Obs.items()}
                        omega_rs = {}
                        opinion = {}
                        for edge in E:
                            omega_rs[edge] = get_omega_obs(t_Obs[edge])
                            opinion[edge] = get_alphabeta(omega_rs[edge])
                        belief = np.zeros(len(E))
                        uncertain = np.zeros(len(E))
                        for i in range(len(E)):
                            belief[i] = omega_rs[E[i]][0]
                            uncertain[i] = omega_rs[E[i]][2]
                        feat_belief.append(belief)
                        feat_uncertain.append(uncertain)
    return feat_belief, feat_uncertain


def get_epinion_data(T):

    pkl_file = open(
        "/network/rit/lab/ceashpc/xujiang/eopinion_data/tmp/nodes-5000-T-38-rate-0.1-testratio-0.8-swaprate-0.1-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    t_Obs = {e: e_Obs[0:T] for e, e_Obs in Obs.items()}
    omega_rs = {}
    opinion = {}
    for edge in E:
        omega_rs[edge] = get_omega_obs(t_Obs[edge])
        opinion[edge] = get_alphabeta(omega_rs[edge])
    belief = np.zeros(len(E))
    uncertain = np.zeros(len(E))
    test_index = []
    for i in range(len(E)):
        belief[i] = omega_rs[E[i]][0]
        uncertain[i] = omega_rs[E[i]][2]
        if E[i] in E_X:
            test_index.append(i)
    return belief, uncertain, test_index


if __name__ == '__main__':
    # omega, opinion, b, u, alphabeta = get_bn_undirect_beijing()
    # np.save("omega_dict_beijing_20130915ref_proc-0.9.npy", alphabeta)
    # np.save("omega_undirect_beijing.npy", omega)
    # np.save("opinion_undirect_beijing.npy", opinion)
    # np.save("belief_undirect_beijing.npy", b)
    # np.save("uncertain_undirect_beijing.npy", u)
    # op = np.load("opinion_undirect_beijing.npy")
    # ome = np.load("omega_undirect_beijing.npy")
    # op_0 = np.mean(op, axis=0)
    # bb = np.load("belief_undirect_beijing.npy")
    # uu = np.load("uncertain_undirect_beijing.npy")
    # bb = np.load("20130915ref_proc-0.9_neigh.npy")
    # for i in range(len(bb)):
    #     nn = bb[i]
    #     c = np.setdiff1d(nn, [i])
    adj = get_ad_matrix_direct()
    np.save("epinion_node500_alledges", adj)
    # print(np.sum(adj))
    # get_epinion_data(11)
    print(1)
