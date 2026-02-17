import numpy as np
import torch
import pickle
import scipy.sparse as sp 
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_scatter import scatter_add

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def map_data(Data):
    s_data = Data[0]
    s_target = Data[1]
    cur_data = []
    cur_target = []
    for i in range(len(s_data)):
        data = s_data[i]
        target = s_target[i]
        if len(data) > 40:
            continue
        cur_data.append(data)
        cur_target.append(target)
    return [cur_data, cur_target]

def handle_data(inputData, sw, opt):

    len_data = []
    
    # 找到最长交互序列
    for nowData in inputData:
        len_data.append(len(nowData))
    # len_data = [len(nowData) for nowData in inputData]
    max_len = max(len_data)

    edge_lens = []
    for item_seq in inputData:
        item_num = len(list(set(item_seq))) # item序列中item的数量
        num_sw = 0
        if opt.sw_edge: #True
            for win_len in sw:
                temp_num = len(item_seq) - win_len + 1
                num_sw += temp_num
        edge_num = num_sw
        if opt.item_edge:
            edge_num += item_num
        edge_lens.append(edge_num)

    max_edge_num = max(edge_lens)

    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]

    return us_pois, us_msks, max_len, opt.hyper_num #max_edge_num

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)
    
from torch import nn
import torch.nn.functional as F

def data_masks(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo
    
class Data(Dataset):
    def __init__(self, data, all_train, opt, n_node, sw=[3,4]):
        self.n_node = n_node
        inputs, mask, max_len, max_edge_num = handle_data(data[0], sw, opt)#inputs是反过来的会话序列。mask是会话序列的mask。max_len是最长会话长度。max_edge_num是最大超边数
        self.raw = np.asarray(data[0],dtype=object)#############
        adj = data_masks(self.raw, n_node) ############
        adjacency = adj.multiply(1.0/adj.sum(axis=0).reshape(1, -1)) ############
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        self.adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        self.inputs = np.asarray(inputs) #[114790,19]
        self.targets = np.asarray(data[1])#[114790]
        self.mask = np.asarray(mask)#[114790,19]
        self.length = len(data[0])#114790
        self.max_len = max_len # max_node_num 19
        self.max_edge_num = max_edge_num  # max_edge_num 70
        self.sw = sw # slice window [2,3,4]
        self.opt = opt

        self.max_number = 0
        dim_image_text = 64
        img_path = './datasets/'+ opt.dataset + '/imgMatrixpca.npy'
        imgWeights = np.load(img_path)
        self.image_embedding = nn.Embedding(n_node, dim_image_text)
        img_pre_weight = np.array(imgWeights)
        self.image_embedding.weight.data[:imgWeights.shape[0]].copy_(torch.from_numpy(img_pre_weight)) #[k,64]

        text_path = './datasets/' + opt.dataset + '/textMatrixpca.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(n_node, dim_image_text)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data[:textWeights.shape[0]].copy_(torch.from_numpy(text_pre_weight))

    def __getitem__(self, index):
        
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        max_n_node = self.max_len
        max_n_edge = self.max_edge_num # max hyperedge num

        node = np.unique(u_input)
        # print('u_input:',u_input)
        # print('node:',node)
        items = node.tolist() + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        
        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        if self.opt.sw_edge:
            for win in self.sw:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        for j in range(i, i+win): # 在窗口内的节点加入同一超边edge_idx
                            if u_input[j] == 0:
                                break
                            rows.append(np.where(node == u_input[j])[0][0])
                            cols.append(edge_idx)
                            vals.append(1.0)
                        edge_idx += 1
            # print('slide_rows:',rows)
            # print('slide_cols:',cols)
        
        edge_idx = 0
        if self.opt.item_edge:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        if u_input[i] == item and i > 0:
                            rows.append(np.where(node == u_input[i-1])[0][0])
                            cols.append(edge_idx)
                            vals.append(2.0)
                    
                    rows.append(np.where(node == item)[0][0])
                    cols.append(edge_idx)
                    vals.append(2.0)
                    
                    edge_idx += 1
                    
            # print('rows:',rows)
            # print('cols:',cols)
            
            # print('\n')
        # intent hyperedges are dynamic generated in layers.py
        u_Hs = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        Hs = np.asarray(u_Hs.todense())
        
        # 构建Image超图
        image_embeddings = self.image_embedding.weight
        #print(len(u_input))#19
        # print(index)#第几个会话
        embeddings = image_embeddings[u_input] # 形状应为 [序列长度, emb_dim] #【19，64】
        cosine_similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)
        threshold = -0.5
        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        if self.opt.sw_edge:
            for win in self.sw:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        row_ = np.where(node == u_input[i])[0][0]
                        for j in range(i, i+win): # 在窗口内的节点加入同一超边edge_idx
                            if u_input[j] == 0:
                                break
                            row = np.where(node == u_input[j])[0][0]
                            if cosine_similarity_matrix[row][row_] > threshold and cosine_similarity_matrix[row][row_]!=0:
                                rows.append(row)
                                cols.append(edge_idx)
                                vals.append(1.0)
                        edge_idx += 1
        
        edge_idx = 0
        if self.opt.item_edge:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        row_ = np.where(node == u_input[i])[0][0]
                        if u_input[i] == item and i > 0:
                            row = np.where(node == u_input[i-1])[0][0]
                            if cosine_similarity_matrix[row][row_] > threshold and cosine_similarity_matrix[row][row_]!=0:
                                rows.append(row)
                                cols.append(edge_idx)
                                vals.append(2.0)
                    edge_idx += 1
        
        Hs_image = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        Hs_image = np.asarray(Hs_image.todense())
        
        # 构建Text超图
        text_embeddings = self.text_embedding.weight
        embeddings = text_embeddings[u_input] # 形状应为 [序列长度, emb_dim]
        cosine_similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=-1)
        threshold = 1
        # H_s shape: (max_n_node, max_n_edge)
        rows = []
        cols = []
        vals = []
        # generate slide window hyperedge
        edge_idx = 0
        if self.opt.sw_edge:
            for win in self.sw:
                for i in range(len(u_input)-win+1):
                    if i+win <= len(u_input):
                        row_ = np.where(node == u_input[i])[0][0]
                        for j in range(i+1, i+win): # 在窗口内的节点加入同一超边edge_idx
                            if u_input[j] == 0:
                                break
                            row = np.where(node == u_input[j])[0][0]
                            if cosine_similarity_matrix[row][row_] > threshold and cosine_similarity_matrix[row][row_]!=0:
                                rows.append(row)
                                cols.append(edge_idx)
                                vals.append(1.0)
                        edge_idx += 1

        edge_idx = 0
        if self.opt.item_edge:
            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        row_ = np.where(node == u_input[i])[0][0]
                        if u_input[i] == item and i > 0:
                            row = np.where(node == u_input[i-1])[0][0]
                            if cosine_similarity_matrix[row][row_] > threshold and cosine_similarity_matrix[row][row_]!=0:
                                rows.append(row)
                                cols.append(edge_idx)
                                vals.append(2.0)
                    edge_idx += 1

        Hs_text = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
        Hs_text = np.asarray(Hs_text.todense())
        
        return [torch.tensor(alias_inputs), torch.tensor(Hs), torch.tensor(items),
                torch.tensor(mask), torch.tensor(target), torch.tensor(u_input),
                torch.tensor(Hs_image), torch.tensor(Hs_text)] #[19],[19,70],[19],[19],[1],[19],[19,70],[19,70]

    def __len__(self):
        return self.length #114790
