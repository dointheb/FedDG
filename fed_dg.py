import torch
import utils
import copy
import Data.pacs as pacs
from torch import nn
from domain_clf import domain_clf
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset,TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment,InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.init as init
import model

class fed_dg:
    def __init__(self,args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}"if torch.cuda.is_available() else "cpu")
        self.test_domain = args.test_domain
        
        #根据命令行的数据集名称返回相应的数据集路径和超参数
        self.root , self.num_clients, self.train_split, self.domains, self.num_classes, self.batch_size, self.lr, self.hyper = utils.get_params(args)
        #根据数据集路径和域名称返回相应的dataloader
        self.train_loaders, self.valid_loaders, self.test_loader = utils.get_dataloader(self.args.dataset,self.root,self.domains,self.test_domain,self.train_split, self.batch_size)
        self.scheduler_state = None
        #定义全局特征提取器和全局分类器
        #全局特征提取器为resnet18的预训练模型
        self.global_feature_extractor = model.ResNetWithMixStyle(mix_layers=['layer1', 'layer2', 'layer3']).to(self.device)
        state_dict = resnet18(pretrained=True).state_dict()
        self.global_feature_extractor.load_state_dict(state_dict, strict=False)

        self.global_classifier = model.OfficeHomeDGClassifier(feature_dim=512, num_classes=self.num_classes).to(self.device)
        self.global_classifier.apply(utils.init_weights)
        

        #定义客户端特征提取器和客户端分类器列表
        self.clients_feature_extractor = []
        self.clients_classifer = []


    #单个客户端的训练
    #传入参数:全局特征提取器,全局分类器,所有客户端的特征提取器,所有客户端的分类器,该客户端所有的训练集dataloader,该客户端所有的验证集dataloader,客户端id
    def train_clients(self,clients_heads,train_loader,valid_loader,client_id):
        
        #复制全局特征提取器和全局分类器
        local_feature_extractor = copy.deepcopy(self.global_feature_extractor)
        local_classifier = copy.deepcopy(self.global_classifier)
        
        #重新初始化优化器和调度器
        optimizer = torch.optim.SGD([{'params':local_feature_extractor.parameters(),'lr':self.lr},
                                            {'params':local_classifier.parameters(),'lr':self.lr}], momentum=0.9, weight_decay=5e-4)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs, eta_min=0.0001)

        #为了能够按照round进行调度,因此需要在每个round开始时加载调度器的状态
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        #定义每个round客户端的损失和准确率
        sum_loss = 0
        sum_acc = 0
        sum_valid_acc = 0

        #保存每个round客户端的特征和标签(其中最后一个round的特征和标签会被用于域分类器的训练)
        features = []
        labels = []

        #定义每个round客户端的样本数,以便服务器端域分类器的训练(域分类器需要所有客户端的总样本数)
        total_sample = 0

        local_feature_extractor.train()
        local_classifier.train()
        for epoch in range(self.args.n_client_epochs):
            
            #定义每个epoch的训练和验证的损失和准确率 及相应的样本数
            train_loss = 0
            train_correct = 0
            train_samples = 0
            valid_correct = 0
            valid_samples = 0
            total_sample = 0

            idx = 0
            #开始训练
            for idx,(data,aug_data,target) in enumerate(train_loader):
                data,aug_data, target = data.to(self.device), aug_data.to(self.device),target.to(self.device)
                
                feature_orig = local_feature_extractor(data)
                output = local_classifier(feature_orig)
                feature_aug = local_feature_extractor(aug_data)
                output_aug = local_classifier(feature_aug)
                
                #获取特征及标签,将其加入到features和labels中
                features.append(feature_orig.detach())
                labels.append(torch.full((feature_orig.size(0),), client_id, dtype=torch.long))

                
                loss1 = torch.nn.CrossEntropyLoss()(output, target)
                loss2 = torch.nn.CrossEntropyLoss()(output_aug, target)
                
                grad_cossim11 = []
                for n,p in local_classifier.named_parameters():
                    grad_orig = torch.autograd.grad([loss1], [p], 
                                        create_graph=True,
                                        only_inputs=True,
                                        allow_unused=False)[0]
                    grad_aug = torch.autograd.grad([loss2], [p], 
                                        create_graph=True,
                                        only_inputs=True,
                                        allow_unused=False)[0]
                    if len(p.shape) > 1:
                            _cossim = F.cosine_similarity(grad_orig, grad_aug, dim=1).mean()
                    else:
                            _cossim = F.cosine_similarity(grad_orig, grad_aug, dim=0)
                    grad_cossim11.append(_cossim)
                grad_cossim1 = torch.stack(grad_cossim11)
                L_intra = (1.0 - grad_cossim1).mean()
                
                for i,client_head in enumerate(clients_heads):
                    grad_cossim_inter = []
                    output_global = client_head(feature_orig)
                    loss_global = torch.nn.CrossEntropyLoss()(output_global, target)
                    for g_p, p in zip(client_head.parameters(), local_classifier.parameters()):
                        grad_inter = torch.autograd.grad([loss_global],
                                            [g_p],
                                            create_graph=True,
                                            only_inputs=True,
                                            allow_unused=False)[0]
                        grad_aug = torch.autograd.grad([loss2],
                                            [p],
                                            create_graph=True,
                                            only_inputs=True,
                                            allow_unused=False)[0]
                        if len(p.shape) > 1:
                            _cossim_inter = F.cosine_similarity(grad_inter, grad_aug, dim=1).mean()
                        else:
                            _cossim_inter = F.cosine_similarity(grad_inter, grad_aug, dim=0)
                        grad_cossim_inter.append(_cossim_inter)
                    grad_cossim_inter = torch.stack(grad_cossim_inter)
                    if i == 0:
                        L_inter = (1.0 - grad_cossim_inter).mean()
                    else:
                        L_inter += (1.0 - grad_cossim_inter).mean()
                L_inter /= (len(clients_heads))
                
            
                optimizer.zero_grad()
                total_loss = 0.5 * (loss1 + loss2) + self.hyper * L_intra + (1-self.hyper) * L_inter
                total_loss.backward()
                optimizer.step()
                

                train_loss += total_loss.item()
                
                train_correct += (output.argmax(1) == target).sum().item() + (output_aug.argmax(1) == target).sum().item()
                train_samples += len(data) * 2

            total_sample += train_samples / 2
            train_acc = train_correct / train_samples
            train_loss /= (idx+1)
            sum_loss += train_loss
            sum_acc += train_acc
            print(
                f"Client #{client_id} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Train Loss: {train_loss} | Train Acc: {train_acc}",
                end="\r",
            )

            #验证 直接用验证集在本地模型上进行验证
            local_feature_extractor.eval()
            local_classifier.eval()
            with torch.no_grad():
                for idx,(data,target) in enumerate(valid_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    feature = local_feature_extractor(data)
                    output = local_classifier(feature)
                    valid_correct += (output.argmax(1) == target).sum().item()
                    valid_samples += len(data)
                valid_acc = valid_correct / valid_samples
                sum_valid_acc += valid_acc
                print(
                    f"Client #{client_id} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Valid Acc: {valid_acc}",
                    end="\r",
                )

        #每个round结束后进行调度器的更新
        scheduler.step()
        #更新此轮调度器的状态 以便下一轮客户端训练时能够按照round进行调度
        self.scheduler_state = scheduler.state_dict()
        #返回本地特征提取器、本地分类器、平均损失、平均训练准确率、平均验证准确率、所有特征、对应标签、客户端总样本数
        return local_feature_extractor, local_classifier, sum_loss/self.args.n_client_epochs, sum_acc/self.args.n_client_epochs, sum_valid_acc/self.args.n_client_epochs,  \
                torch.cat(features), torch.cat(labels), total_sample
    
    #训练服务器
    def train_server(self):
        
        #初始化各个客户端分类器为全局分类器,以便首次客户端训练
        clients_heads = [copy.deepcopy(self.global_classifier) for _ in range(self.num_clients)]

        #开始训练,一共n_epochs个round
        for i in range(self.args.n_epochs):
            print(f"round {i+1}/{self.args.n_epochs}")

            #用于保存各个客户端的特征提取器和分类器,以便服务器聚合使用
            self.clients_feature_extractor = []
            self.clients_classifer = []

            clients_losses = []
            client_accs = []
            client_valid_accs = []

            #用于保存各个客户端的特征和相应标签,以此来训练域分类器(每个round训练一次)
            client_features = []
            client_labels = []

            #作为中间变量给clients_heads传值
            temp = []

            #用于保存各个客户端的样本数
            client_samples = []
            
            #所有客户端的样本数
            total_sample = 0


            #对每个客户端进行训练
            for client_id in range(self.num_clients):
                #返回客户端特征提取器、客户端分类器、平均损失、平均训练准确率、平均验证准确率、所有特征、对应标签、客户端总样本数
                client_feature_extractor, client_classifier, client_loss ,client_acc, client_valid_acc, client_feature, client_label, client_sample= self.train_clients(
                                                                                             clients_heads,
                                                                                             self.train_loaders[client_id],
                                                                                             self.valid_loaders[client_id],
                                                                                             client_id)
                
                #保存客户端特征提取器和分类器
                self.clients_feature_extractor.append(client_feature_extractor.state_dict())
                self.clients_classifer.append(client_classifier.state_dict())

                #保存分类器
                temp.append(client_classifier)

                clients_losses.append(client_loss)
                client_accs.append(client_acc)
                client_valid_accs.append(client_valid_acc)

                #保存特征和标签 以便域分类器的训练
                client_features.append(client_feature)
                client_labels.append(client_label)

                client_samples.append(client_sample)
                total_sample += client_sample
            
            #下一轮round发给客户端的分类器 使用上一轮各个客户端的分类器
            clients_heads = temp

            #对客户端的特征和标签进行加权平均,更新全局特征提取器和全局分类器
            updated_weights_f = utils.static_avg(self.clients_feature_extractor,client_samples)
            updated_weights_c = utils.static_avg(self.clients_classifer,client_samples)
            self.global_feature_extractor.load_state_dict(updated_weights_f)
            self.global_classifier.load_state_dict(updated_weights_c)

            avg_loss = sum(clients_losses) / len(clients_losses)
            avg_acc = sum(client_accs) / len(client_accs)
            avg_valid_acc = sum(client_valid_accs) / len(client_valid_accs)
            print(f"round {i+1}/{self.args.n_epochs} | Average Loss: {avg_loss} | Average train Acc: {avg_acc} | Average valid Acc: {avg_valid_acc}")

            # #将所有客户端的特征和标签拼接起来
            # client_features = torch.cat(client_features)
            # client_labels = torch.cat(client_labels)

            # #将所有客户端的特征和标签转换为dataset和dataloader
            # fe_dataset = TensorDataset(client_features,client_labels)
            # fe_dataloader = DataLoader(fe_dataset, batch_size=self.batch_size, shuffle=True)

            # #防止重复初始化域分类器
            # if i == 0:
            #     self.domain_classifier = domain_clf(512,self.num_clients,int(total_sample)).to(self.device)
            # optimizer = torch.optim.SGD(self.domain_classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

            # fe_loss = 0
            # fe_correct = 0
            # idx = 0

            # #训练域分类器
            # for idx,(data,target) in enumerate(fe_dataloader):
            #     data, target = data.to(self.device), target.to(self.device)
            #     data = data.unsqueeze(-1).unsqueeze(-1)
            #     optimizer.zero_grad()
            #     output = self.domain_classifier(data)
            #     loss = torch.nn.CrossEntropyLoss()(output, target)
            #     loss.backward()
            #     optimizer.step()
            #     fe_loss += loss.item()
            #     fe_correct += (output.argmax(-1) == target).sum().item()
            
            # print(f"round {i+1}/{self.args.n_epochs} | Classifier Loss: {fe_loss / (idx+1)} | Classifier Acc: {fe_correct/total_sample}")

            
            

    def test(self):

        #定义新的全局特征提取器和全局分类器(动态聚合)
        # model = models.resnet18(pretrained=True).to(self.device)
        # model.fc = nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes)
        
        # new_global_feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        # new_global_classifer = nn.Linear(in_features=model.fc.in_features, out_features=self.num_classes).to(self.device)
        
        self.global_feature_extractor.eval()
        self.global_classifier.eval()
        test_correct = 0
        test_sample = 0
        pbar = tqdm(self.test_loader, ncols=110)

        for idx, (data, target) in enumerate(pbar):
            
            # # 将测试数据输入到全局特征提取器,获得特征后加入域分类器进行类别预测,得到域概率分布output
            # data, target = data.to(self.device), target.to(self.device)
            # feature = self.global_feature_extractor(data)
            # output = self.domain_classifier(feature.unsqueeze(-1).unsqueeze(-1))
            # output = output.squeeze(0)

            # # 使用域概率分布output作为权重参数 动态聚合客户端特征提取器和分类器,获得新的全局特征提取器和全局分类器
            # new_updated_weights_f = utils.dynamic_avg(self.clients_feature_extractor,output)
            # new_updated_weights_c = utils.dynamic_avg(self.clients_classifer,output)
            # new_global_feature_extractor.load_state_dict(new_updated_weights_f)
            # new_global_classifer.load_state_dict(new_updated_weights_c)
            
            # # 将新的全局特征提取器和全局分类器用于测试
            # feature = new_global_feature_extractor(data).flatten(1)
            # output = new_global_classifer(feature)

            # 不采用test-time
            data, target = data.to(self.device), target.to(self.device)
            feature = self.global_feature_extractor(data)
            output = self.global_classifier(feature)

            test_correct += (output.argmax(1) == target).sum().item()
            test_sample += len(data)

        test_acc = test_correct / test_sample
        print(f"Test Accuracy: {test_acc}") 

        
    
if __name__ == "__main__":
    args = utils.arg_parser()
    fed = fed_dg(args)
    fed.train_server()
    fed.test()
    
        
        

        

                
        

    
    





        
        

