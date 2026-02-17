import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from layers import *
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from utils import *

# 这个是有GNN的代码，有GNN的组合可能在不同数据集上效果比单独超图更好一些，具体见语雀。
        
class MODEL(Module):
    def __init__(self, adj, opt, num_node, adj_all=None, num=None, cat=False):
        super(MODEL, self).__init__()
        # HYPER PARA
        self.opt = opt 
        self.batch_size = opt.batch_size #64
        self.num_node = num_node #11638
        self.dim = opt.hiddenSize #200
        self.layer = int(opt.layer) #1
        self.ssl_temp = opt.temperature #0.1
        self.knn_k = opt.knn_k #5
        self.n_layers = 1
        self.sparse = True
        self.k = opt.rank 
        self.mlp1 = MLP(self.dim,self.dim*self.k,self.dim//2,self.dim*self.k)
        self.mlp2 = MLP(self.dim,self.dim*self.k,self.dim//2,self.dim*self.k)
        self.meta_net = nn.Linear(self.dim*3, self.dim, bias=True)
        self.auxi = opt.auxi
        self.dropout = nn.Dropout(0.5)
        self.adj = adj
        
        # Item representation
        self.embedding = nn.Embedding(num_node, self.dim) 
        self.feat_latent_dim = self.dim
        
        # Position representation
        self.pos_embedding = nn.Embedding(200, self.dim)
        self.hyper_agg = LocalHyperGATlayer(self.dim, self.layer, self.opt.alpha)
        self.hyper_agg_image = LocalHyperGATlayer(self.dim, self.layer, self.opt.alpha)
        self.hyper_agg_text = LocalHyperGATlayer(self.dim, self.layer, self.opt.alpha)
        self.gnn_agg = SimpleGATLayer(self.dim, self.layer, self.opt.alpha)
        self.gnn_agg_image = SimpleGATLayer(self.dim, self.layer, self.opt.alpha)
        self.gnn_agg_text = SimpleGATLayer(self.dim, self.layer, self.opt.alpha)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim)) #400，200
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)
         
        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.linear = nn.Linear(2 * self.dim, self.dim)
        # main task loss
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

        dim_image_text = 64
        # introducing text&image embeddings
        img_path = './datasets/'+ opt.dataset + '/imgMatrixpca.npy'
        imgWeights = np.load(img_path)
        self.image_embedding = nn.Embedding(num_node, dim_image_text)
        img_pre_weight = np.array(imgWeights)
        self.image_embedding.weight.data[:imgWeights.shape[0]].copy_(torch.from_numpy(img_pre_weight))

        text_path = './datasets/' + opt.dataset + '/textMatrixpca.npy'
        textWeights = np.load(text_path)
        self.text_embedding = nn.Embedding(num_node, dim_image_text)
        text_pre_weight = np.array(textWeights)
        self.text_embedding.weight.data[:textWeights.shape[0]].copy_(torch.from_numpy(text_pre_weight))


        # 多模态相似度
        self.image_original_adj, self.text_original_adj = self.auxiliary_task()

        # 模态融合层
        self.fusion_fc = nn.Linear(dim_image_text+self.dim, self.dim)
        self.image_to_id = nn.Linear(dim_image_text, self.dim)
        self.text_to_id = nn.Linear(dim_image_text, self.dim)

        self.image_trs = nn.Linear(dim_image_text, self.dim)
        self.text_trs = nn.Linear(dim_image_text, self.dim)
        self.gate_v = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid()
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def auxiliary_task(self):#, item_emb, image_emb, text_emb):
        '''
        use the knowledge from modality view to enhance the item embeddings of collaborative view
        '''
        #计算模态相似度, 选取最相似的模态物品topk
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        image_original_adj = image_adj.cuda()
        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        text_original_adj = text_adj.cuda()

        return image_original_adj, text_original_adj


    def metatransform(self, auxiembedi, targetembedi, ineighbor):
        
        # Neighbor information of the target node
        # ineighbor=torch.matmul( self.iuadj.cuda(),self.ui_userEmbedding)

        # Meta-knowlege extraction         #[64,19,200]->[64,19,600]->[64,19,200]
        lat = torch.sum(torch.cat([auxiembedi, targetembedi, ineighbor], dim=-1),dim=1)#[64,600]
        tembedi=(self.meta_net(lat.detach()))#[64,200]
        
        
        """ Personalized transformation parameter matrix """
        # Low rank matrix decomposition
        metai1=self.mlp1(tembedi).reshape(-1,self.dim,self.k)# d*k #[64,600]->[64,200,3]
        metai2=self.mlp2(tembedi).reshape(-1,self.k,self.dim)# k*d #[64,600]->[64,3,200]
        
        meta_biasi =(torch.mean( metai1,dim=0))#[200,3]
        meta_biasi1=(torch.mean( metai2,dim=0))#[3,200]
        
        low_weighti1=F.softmax( metai1 + meta_biasi, dim=1)#[64,200,3]
        low_weighti2=F.softmax( metai2 + meta_biasi1,dim=1)#[64,3,200]

        # The learned matrix as the weights of the transformed network
                                #[64,19,200],[64,200,3]->[64,19,3]
        tembedis = torch.matmul(auxiembedi, low_weighti1)# Equal to a two-layer linear network;
        tembedis = torch.matmul(tembedis, low_weighti2)##[64,19,3],[64,3,200]->[64,19,200]
        
        return  tembedis#[64,19,200]

    # cross view contrastive learning
    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss
    
    def enhanced_info_nce_loss(self, augmented, original, margin=0.1):
        """
        困难样本增强的对比学习损失函数（InfoNCE改进版）

        参数：
        - augmented: 增强后的样本张量，形状为 (batch_size, feature_dim)
        - original: 原始样本张量，形状为 (batch_size, feature_dim)
        - temperature: 温度参数，用于控制对比学习的平滑程度
        - margin: 困难样本增强的边界，用于调节困难样本的权重

        返回：
        - loss: 计算的对比学习损失值
        """
        batch_size = augmented.size(0)

        # 归一化向量
        augmented_norm = F.normalize(augmented, dim=1)
        original_norm = F.normalize(original, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(augmented_norm, original_norm.T) / self.ssl_temp

        # 生成标签，正例的标签是对角线上的
        labels = torch.arange(batch_size).to(original.device)

        # 提取正例相似度 (对角线上的值)
        positive_similarities = similarity_matrix[torch.arange(batch_size), labels]

        # 提取负例相似度 (非对角线上的值)
        negative_similarities = similarity_matrix + torch.eye(batch_size).to(original.device) * -float('inf')

        # 困难样本增强逻辑
        # 选择困难正例（正例相似度低于 margin 的样本）
        hard_positive_mask = positive_similarities < margin
        hard_positive_loss = -torch.log(torch.exp(positive_similarities) + 1e-10) # 避免数值问题
        hard_positive_loss = hard_positive_loss[hard_positive_mask].mean() if hard_positive_mask.any() else 0.0

        # 选择困难负例（负例相似度高于 margin 的样本）
        hard_negative_mask = negative_similarities > margin
        hard_negative_loss = torch.logsumexp(negative_similarities[hard_negative_mask], dim=0) if hard_negative_mask.any() else 0.0

        # 汇总损失：困难正例损失 + 困难负例损失
        loss = hard_positive_loss + hard_negative_loss

        return loss

    def cal_loss_cl(self, hg_item_embs, hg_image_embs, hg_text_embs):

        # normalization
        norm_item_embs = F.normalize(hg_item_embs, p=2, dim=1)
        norm_image_embs = F.normalize(hg_image_embs, p=2, dim=1)
        norm_text_embs = F.normalize(hg_text_embs, p=2, dim=1)

        # calculate loss
        loss_cl = 0.0
        loss_cl += self.enhanced_info_nce_loss(norm_image_embs, norm_item_embs)
        loss_cl += self.enhanced_info_nce_loss(norm_text_embs, norm_item_embs)
        loss_cl += self.enhanced_info_nce_loss(norm_image_embs, norm_text_embs)
        
        # loss_cl += self.cal_loss_infonce(norm_image_embs, norm_item_embs) # 
        # loss_cl += self.cal_loss_infonce(norm_text_embs, norm_item_embs)
        # loss_cl += self.cal_loss_infonce(norm_image_embs, norm_text_embs)

        return loss_cl

    def metaknowledge(self, lat1, lat2):
        lat = torch.cat([lat1, lat2], dim=-1)
        lat = self.leakyrelu(self.dropout(self.linear(lat))) + lat1 + lat2
        return lat

    def compute_scores(self, hidden, mask, item_embeddings):
        mask = mask.float().unsqueeze(-1)#[64,19,1]

        batch_size = hidden.shape[0]#64
        len = hidden.shape[1]#19
        pos_emb = self.pos_embedding.weight[:len]#[19,200]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)#[64,19,200]

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)#[64,200]
        hs = hs.unsqueeze(-2).repeat(1, len, 1) #[64,19,200]
        ht = hidden[:, 0, :] #[64,200]
        ht = ht.unsqueeze(-2).repeat(1, len, 1)#[64,19,200]             # (b, N, dim)
        
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        

        hs = torch.cat([hs, ht], -1).matmul(self.w_s)

        feat = hs * hidden  
        nh = torch.sigmoid(torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1))

        # nh = torch.sigmoid(self.glu1(nh))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        b = item_embeddings[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))

        return scores ##[64,11638]

    def multimodal_fusion(self, id_embeddings, image_embeddings, text_embeddings, mask_item): 
        """
        基于 ID 置信度融合 ID、文本和图像嵌入。

        Args:
        id_embeddings: ID 嵌入 (batchsize, 19, 200).
        text_embeddings: 文本嵌入 (batchsize, 19, 200).
        image_embeddings: 图像嵌入 (batchsize, 19, 200).
        mask_item: ID 置信度掩码 (batchsize, 19).

        Returns:
        融合后的嵌入 (batchsize, 19, 200).
        """
        
        batch_size, seq_len, embedding_dim = id_embeddings.shape

        # 将 mask_item 转换为浮点型
        mask_item = mask_item.float()

        # 计算有效 ID 的数量
        num_valid_ids = mask_item.sum(dim=1, keepdim=True) # (batchsize, 1) 有效ID的数量

        # 计算全局的有效ID数量的平均值和标准差
        mean_valid_ids = num_valid_ids.mean() # 有效ID数量的平均值
        std_valid_ids = num_valid_ids.std(unbiased=False) + 1e-6 # 标准差，并保证数值稳定性

        # 使用 Sigmoid 函数来控制 id_confidence，并通过动态调整中心点和增长速率
        # 这里的 sigmoid 函数可以让 id_confidence 随着 num_valid_ids 增大而增大
        growth_rate = 5.0 # 控制 Sigmoid 函数的增长速率，值越大，增长越快
        offset = mean_valid_ids # 以平均值为中心，动态调整 Sigmoid 的偏移量
        # Sigmoid 映射：num_valid_ids 越大，id_confidence 越接近 1
        id_confidence = torch.sigmoid(growth_rate * (num_valid_ids - offset) / (std_valid_ids + 1e-6))

        # 将置信度归一化到 [0, 1]
        id_confidence = (id_confidence - id_confidence.min(dim=0, keepdim=True).values) / \
        (id_confidence.max(dim=0, keepdim=True).values - id_confidence.min(dim=0, keepdim=True).values + 1e-6) #[64,1]

        # 扩展置信度以便广播
        id_confidence = id_confidence.unsqueeze(2).expand(-1, -1, embedding_dim) # (batchsize, 1, 200)

        # 自适应融合：基于 ID 置信度的加权平均
        # 当 id_confidence 趋近于 1 时，更多地依赖 id_embeddings，反之更多依赖文本和图像嵌入
        fused_embeddings = id_confidence * id_embeddings + (1 - id_confidence) * (text_embeddings + image_embeddings) / 2  # 

        return fused_embeddings
        
    def forward(self, inputs, Hs, mask_item, item, Hs_image, Hs_text):#, Hs_image, Hs_text):
                      #[64,19],[64,19,70],[64,19],[64,19],[64,19,70],[64,19,70]

        # 图像和文本表征
        Hs_image = Hs_image.to(Hs.device)## [64,19,70]
        Hs_image = Hs_image.type(Hs.dtype)
        Hs_text = Hs_text.to(Hs.device)
        Hs_text = Hs_text.type(Hs.dtype)
        
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
    
        # ID超图
        item_embeddings = self.embedding.weight##[11638,200]

        # # 用ID净化模态特征
        image_feats = self.image_trs(self.image_embedding.weight)## [11638,200]
        text_feats = self.text_trs(self.text_embedding.weight)
        image_emb = torch.multiply(self.embedding.weight, self.gate_v(image_feats)) ## [11638,200]
        text_emb = torch.multiply(self.embedding.weight, self.gate_t(text_feats))
        # 图像和文本表征进一步过 GCN 聚合
        image_item_embeds = torch.sparse.mm(self.image_original_adj, image_emb) ## [11638,200]
        text_item_embeds = torch.sparse.mm(self.text_original_adj, text_emb) # torch.zeros(image_item_embeds.shape).to(image_item_embeds.device) #
        # 融合
        item_embeddings = item_embeddings + self.metaknowledge(image_item_embeds, text_item_embeds) #[11638,200]

        """ 3 个模态的超图 """
        
        ### ID 
        zeros = trans_to_cuda(torch.FloatTensor(1, self.dim).fill_(0))
        item_embeddings = torch.cat([zeros, item_embeddings], 0) ##[11638,200]
        h = item_embeddings[inputs]#[64,19,200]
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(-1)##[64,19,200]
        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)##[64,200]
        session_c = session_c.unsqueeze(1)  # (batchsize, 1, dim) #[64,1,200]
        h_local = self.hyper_agg(h, Hs, session_c) ##[64,19,200]
        # h_gnn = self.gnn_agg(h, Hs, session_c) ##[64,19,200]

        ### 图像
        image_embeddings = torch.cat([zeros, image_emb], 0)
        h_image = image_embeddings[inputs]#[64,19,200]
        image_emb = image_embeddings[item] * mask_item.float().unsqueeze(-1)#[64,19,200]
        session_c_image = torch.sum(image_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)#[64,200]
        session_c_image = session_c_image.unsqueeze(1)  # (batchsize, 1, dim)
        h_local_image = self.hyper_agg_image(h_image, Hs_image, session_c_image)#[64,19,200]
        # h_gnn_image = self.gnn_agg_image(h_image, Hs_image, session_c_image)

        ### 文本 
        text_embeddings = torch.cat([zeros, text_emb], 0)
        h_text = text_embeddings[inputs]
        text_emb = text_embeddings[item] * mask_item.float().unsqueeze(-1)
        session_c_text = torch.sum(text_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        session_c_text = session_c_text.unsqueeze(1)  # (batchsize, 1, dim)
        h_local_text = self.hyper_agg_text(h_text, Hs_text, session_c_text)
        # h_gnn_text = self.gnn_agg_text(h_text, Hs_text, session_c_text)
        

        # 3 个模态超图的结果进行聚合
        meta_image = self.metatransform(h_local_image, h_local, session_c_image.repeat(1, h_local.shape[1], 1))#[64,19,200]
        meta_text = self.metatransform(h_local_text, h_local, session_c_text.repeat(1, h_local.shape[1], 1))
        output = h_local + self.metaknowledge(meta_image, meta_text) + self.multimodal_fusion(h_local, meta_image, meta_text, mask_item)
        output = F.normalize(output, p=2, dim=-1) ##[64,19,200]

        # meta_image_gnn = self.metatransform(h_gnn_image, h_gnn, session_c_image.repeat(1, h_gnn.shape[1], 1))#[64,19,200]
        # meta_text_gnn = self.metatransform(h_gnn_text, h_gnn, session_c_text.repeat(1, h_gnn.shape[1], 1))
        # gnn_output = h_gnn + self.metaknowledge(meta_image, meta_text) + self.multimodal_fusion(h_gnn, meta_image, meta_text, mask_item)
        # gnn_output = F.normalize(output, p=2, dim=-1)

        # 超图视图 的 3 个模态对比学习
        loss_cl = self.cal_loss_cl(torch.sum(h_local, 1), torch.sum(h_local_image, 1), torch.sum(h_local_text, 1))
        # loss_gnn_cl = self.cal_loss_cl(torch.sum(h_gnn, 1), torch.sum(h_gnn_image, 1), torch.sum(h_gnn_text, 1))

        return output, item_embeddings, loss_cl#[64,19,200], [11639,200], scalar
        # 这里output在有些数据集上替换成output+gnn_output能达到最好效果 比如我实验记录的用超图的output , 超图和GNN的loss这个组合在Grocery上能达到最好效果
        # loss_cl在有些数据集上替换成loss_cl + loss_gnn_cl能达到最好效果

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre =   nn.Linear(input_dim, feature_dim,bias=True)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out =    nn.Linear(feature_dim, output_dim,bias=True)

    def forward(self, data):
        x = data
        if self.feature_pre:
            x = self.linear_pre(x)
        prelu=nn.PReLU().cuda()
        x = prelu(x) 
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.tanh(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable.cpu()

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, data):
    alias_inputs, Hs, items, mask, targets, inputs, Hs_image, Hs_text = data # , Hs_image, Hs_text
    alias_inputs = trans_to_cuda(alias_inputs).long()#【64，19】
    items = trans_to_cuda(items).long()#[64,19]
    Hs = trans_to_cuda(Hs).float()#[64,19,70]
    mask = trans_to_cuda(mask).long()#[64,19]
    inputs = trans_to_cuda(inputs).long()#[64,19]

    hidden, item_embeddings, loss_cl = model(items, Hs, mask, inputs, Hs_image, Hs_text) # , Hs_image, Hs_text #[64,19,200], [11639,200]
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])#[batch, max_session_len, hidden_size]#[64,19,200]
    # cross view contrastive learning
    return targets, model.compute_scores(seq_hidden, mask, item_embeddings), loss_cl#[64],[64,11638]


def train_test(model, train_data, test_data, top_K, opt):
    #print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()#梯度清零
        targets, scores, loss_cl = forward(model, data)#[64],[64,11638]
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1) + opt.lambda_cl * loss_cl

        loss.backward()
        model.optimizer.step()
        total_loss += loss
    model.scheduler.step()

    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    
    for data in tqdm(test_loader):
        targets, scores, loss_cl = forward(model, data)
        targets = targets.numpy()
        for K in top_K:
            sub_scores = scores.topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(score == target - 1)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(score == target - 1)[0][0] + 2)))
    
    return metrics