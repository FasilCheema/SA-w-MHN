'''
Authors: Fasil Cheema and Mahmoud Alsaeed 
Purpose: This module is used to define a class that handles everything to do with random walks
'''

class RandomWalkGenerator():
    def MultiRankAlgo(self, A_h, A_k, A_m, B_hk, B_hm, B_kh, B_km, B_mh, B_mk):
        ''' The algorithm takes the adjacency matrices of each layer as well as the bipartite adjacency matrices. 
        '''
        num_h = A_h.shape[0]
        num_k = A_k.shape[0]
        num_m = A_m.shape[0]

        W_h = 0
        W_k = 0
        W_m = 0

        for i in range(num_h):
            for j in range(num_h):
                W_h = A_h[i,j] + W_h
        
        for i in range(num_k):
            for j in range(num_k):
                W_k = A_k[i,j] + W_k

        for i in range(num_m):
            for j in range(num_m):
                W_m = A_m[i,j] + W_m

    def MultiRank(H):
        s = 1; a = 0
        if elite:
            s = -1
        X = dict(); Z = dict()
        for ig in gamma_range:
            gamma = 0.1 * ig
            x, z = MultiRank_Nodes_Layers(H, r, gamma, s, a)
            X[ig]=[i for i in sorted(enumerate(x.tolist()), key=lambda x:x[1], reverse=True)]
            Z[ig]=[i for i in sorted(enumerate(z.tolist()), key=lambda x:x[1], reverse=True)]

        X_list = dict.fromkeys(X, [0] * H.total_nodes)
        Z_list = dict.fromkeys(X, [0] * H.num_layers)

        for ig in gamma_range:
            for k, v in Z.items():
                for item in v:
                    layer_idx = item[0]
                    influence = item[1][0]
                    Z_list[ig][layer_idx] = influence
            Z_list[ig] = np.divide(np.array(Z_list[ig]), np.array(Z_list[ig]).sum())

            for k, v in X.items():
                for item in v:
                    node_idx = item[0]
                    influence = item[1][0]
                    X_list[ig][node_idx] = influence
            X_list[ig] = np.divide(np.array(X_list[ig]), np.array(X_list[ig]).sum())

        return X_list, Z_lis
        

        return None 
    
    def RWGenerator(self, delta, prob_matrix, pi_t, pi_0, eta_h, eta_m, eta_k):
        ''' Given the probability matrix, the previous iteration of pi, and pi_rs (a constant) we compute the next iteration of pi 
        '''
        pi_rs = np.zeros((3,1))

        pi_rs[0,0] = eta_h*pi_0h
        pi_rs[0,1] = eta_m*pi_0m
        pi_rs[0,2] = eta_k*pi_0k

        pi_new = (1-delta)*np.matmul(prob_matrix,pi_t) + delta*pi_rs

        return None