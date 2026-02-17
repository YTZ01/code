import sys
import time
import argparse
import pickle
import os
import logging
from sessionG1 import *
from utils import *

save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CDs_and_Vinyl', help='CDs_and_Vinyl/Grocery_and_Gourmet_Food/Office_products/')
parser.add_argument('--model', default='MODEL', help='[GCEGNN, SRGNN, DHCN, SAHNN, COTREC]')
parser.add_argument('--hiddenSize', type=int, default=200)  # best is 200
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--w', type=int, default=4, help='max window size')
parser.add_argument('--gpu_id', type=str,default="0")
parser.add_argument('--batch_size', type=int, default=64)  # best is 64
parser.add_argument('--lr', type=float, default=0.002, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--layer', type=int, default=1, help='the number of layer used')
parser.add_argument('--n_iter', type=int, default=1)  
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')     # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--seed', type=int, default=1234)                                 # [1, 2]
parser.add_argument('--sw_edge', default=True, help='slide_window_edge')
parser.add_argument('--item_edge', default=True, help='item_edge')
parser.add_argument('--validation', default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--knn_k', type=float, default=5)
parser.add_argument('--n_layers', type=float, default=1)
parser.add_argument('--auxi', type=float, default=1)
parser.add_argument('--rank', type = int, default=3, help='the dimension of low rank matrix decomposition') 
parser.add_argument('--hyper_num', type=int, default=70)  # 70, 80, 90, 100
parser.add_argument('--temperature', type=float, default=0.3)# 0.5, 0.4, 0.3, 0.2, 0.1
parser.add_argument('--lambda_cl', type=float, default=0.01, help='lambda of contrastive loss')# 0.001,0.01,0.1

opt = parser.parse_args()
# print("opt is: ", opt)


os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    exp_seed = opt.seed
    top_K = [5, 10, 20]
    init_seed(exp_seed)

    sw = []
    for i in range(2, opt.w+1): 
        sw.append(i) #[2,3,4]

    
    if opt.dataset == 'Tmall':
        num_node = 40727
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.0
        opt.e = 0.4
        opt.w = 6
        # opt.nonhybrid = True
        sw = []
        for i in range(2, opt.w+1):
            sw.append(i)

    elif opt.dataset == 'lastfm':
        num_node = 35231
        opt.n_iter = 1
        opt.dropout_gcn = 0.1
        opt.dropout_local = 0.0

    elif opt.dataset == 'Grocery_and_Gourmet_Food':
        num_node = 11638
        opt.n_iter = 1
    
    elif opt.dataset == 'Cell_Phones_and_Accessories':
        num_node = 8615
        opt.n_iter = 1

    elif opt.dataset == 'Sports_and_Outdoors':
        num_node = 18797
        opt.n_iter = 1

    elif opt.dataset == 'CDs_and_Vinyl':
        num_node = 9527
        opt.n_iter = 1

    elif opt.dataset == 'Office_Products':
        num_node = 9130
        opt.n_iter = 1
        

    else:
        num_node = 310

    # print(">>SEED:{}".format(exp_seed))
    # # ==============================
    # print('===========config================')
    # print("model:{}".format(opt.model))
    # print("dataset:{}".format(opt.dataset))
    # print("gpu:{}".format(opt.gpu_id))
    # print("item_edge:{}".format(opt.item_edge))
    # print("sw_edge:{}".format(opt.sw_edge))
    # print("Test Topks{}:".format(top_K))
    # print(f"Slide Window:{sw}")
    # print('===========end===================')
   
    datapath = r'./datasets/'
    all_train = pickle.load(open(datapath + opt.dataset + '/new_train.txt', 'rb'))
    train_data = pickle.load(open(datapath + opt.dataset + '/new_train.txt', 'rb'))

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(datapath + opt.dataset + '/new_test.txt', 'rb'))

    train_data = Data(train_data, all_train, opt, n_node=num_node, sw=sw)
    # print(train_data.max_number)
    test_data = Data(test_data, all_train, opt, n_node=num_node, sw=sw)

    if opt.model == 'MODEL':
        model = trans_to_cuda(MODEL(train_data.adjacency, opt, num_node))
        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %d' % (total))
    start = time.time()

    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0, 0]
        best_results['metric%d' % K] = [0, 0, 0]

    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print(f'EPOCH:{epoch}')
        print(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
        metrics = train_test(model, train_data, test_data, top_K, opt)



        # print(train_data.max_number)
        for K in top_K:
            flag = 0  # 初始化flag为0
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            metrics['ndcg%d' % K] = np.mean(metrics['ndcg%d' % K]) * 100

            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag += 1
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag += 1
            if best_results['metric%d' % K][2] < metrics['ndcg%d' % K]:
                best_results['metric%d' % K][2] = metrics['ndcg%d' % K]
                best_results['epoch%d' % K][2] = epoch
                flag += 1

            if flag>=2 and K==20:
                # 保存模型（以Hit@K为准，也可根据需要修改判断条件）
                save_path = f'{save_dir}/{opt.model}_{opt.dataset}_best_epoch{epoch}_hit{K}.pth'
                torch.save(model.state_dict(), save_path)
                print(f"Best Hit@{K} model saved to {save_path}")

        for K in top_K:
            print('Current Result:')
            print('\tP@%d: %.4f\tMRR@%d: %.4f\tNDCG@%d: %.4f' %
                (K, metrics['hit%d' % K], K, metrics['mrr%d' % K], K, metrics['ndcg%d' % K]))
            print('Best Result:')
            print('\tP@%d: %.4f\tMRR@%d: %.4f\tNDCG@%d: %.4f\tEpoch: %d, %d, %d' %
                (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1], K, best_results['metric%d' % K][2], 
                 best_results['epoch%d' % K][0], best_results['epoch%d' % K][1], best_results['epoch%d' % K][2]))
            bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
