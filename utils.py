import torch
from typing import  Dict, List
import argparse

#根据客户端的样本数对模型参数进行加权平均
def static_avg(weights:List[Dict[str,torch.Tensor]],client_samples) -> Dict[str,torch.Tensor]:

    total_samples = sum(client_samples)
    weights_avg = {key: torch.zeros_like(weights[0][key]) for key in weights[0].keys()}

    for i, model_weights in enumerate(weights):
        for key in weights_avg.keys():
            weights_avg[key] = weights_avg[key].to(torch.float32)
            weights_avg[key] += model_weights[key] * (client_samples[i] / total_samples)
    
    return weights_avg

#根据客户端的分类结果对模型参数进行加权平均
def dynamic_avg(weights:List[Dict[str,torch.Tensor]],cls_res:torch.Tensor) -> Dict[str,torch.Tensor]:
    cls_weights = cls_res.view(-1)
    weights_avg = {key: torch.zeros_like(weights[0][key]) for key in weights[0].keys()}
    for i, model_weights in enumerate(weights):
        for key in weights_avg.keys():
            weights_avg[key] = weights_avg[key].to(torch.float32)
            weights_avg[key] += model_weights[key] * cls_weights[i]

    return weights_avg

#根据命令行的参数设置数据集的路径和超参数
def get_params(args):
    path, num_clients, train_split, domains, num_classes, batch_size, lr, hyper= None, 3, None, None, 1, None, 0.001, 0.3
    if args.dataset == 'PACS':
        path = '/newdata3/nzw/Datasets/PACS/images'
        num_clients = 3
        train_split = 0.8
        domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        num_classes = 7
        batch_size = 16
        lr = 0.001
        hyper = 0.3

    elif args.dataset == 'OfficeHome':
        path = '/newdata3/wsj/OfficeHome'
        num_clients = 3
        train_split = 0.9
        domains = ['Art', 'Clipart', 'Product', 'Real World']
        num_classes = 65
        batch_size = 30
        lr = 0.002
        hyper = 0.8

    elif args.dataset == 'VLCS':
        path = '/newdata3/nzw/Datasets/VLCS'
        num_clients = 3
        train_split = 0.8
        domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
        num_classes = 5
        batch_size = 16
        lr = 0.001
        hyper = 0.8
        
    elif args.dataset == 'DomainNet':
        path = '/newdata3/wsj/DomainNet'
        num_clients = 5
        train_split = 0.7
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real','sketch']
        num_classes = 345
        batch_size = 16
        lr = 0.001
        hyper = 0.8
    
    return path, num_clients, train_split, domains, num_classes, batch_size, lr, hyper

#设置命令行参数
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PACS', help='dataset name')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--test_domain", type=str, default='photo')

    parser.add_argument("--n_epochs", type=int, default=40)
    parser.add_argument("--n_client_epochs", type=int, default=1)

    return parser.parse_args()