import torch
import utils
import copy
from torch import nn
from domain_clf import domain_clf
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, Subset,TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm

class fed_dg:
    def __init__(self,args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device}"if torch.cuda.is_available() else "cpu")
        self.test_domain = args.test_domain
        
        self.root , self.num_clients, self.train_split, self.domains, self.num_classes, self.batch_size, self.lr, self.hyper = utils.get_params(args)
        self.train_loaders, self.valid_loaders, self.test_loader = self.get_data(self.root,self.domains,self.test_domain,self.train_split)
        
        self.global_feature_extractor = models.resnet18(pretrained=True).to(self.device)
        self.global_feature_extractor.fc = nn.Linear(512, 512).to(self.device)
        self.global_classifer = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        ).to(self.device)

        self.clients_feature_extractor = []
        self.clients_classifer = []

        self.domain_classifier = domain_clf(512,self.num_clients,int(10)).to(self.device)
        self.new_global_feature_extractor = models.resnet18(pretrained=True).to(self.device)
        self.new_global_feature_extractor.fc = nn.Linear(512, 512).to(self.device)
        self.new_global_classifer = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        ).to(self.device)

    def load_domain_data(self,root,domain):
        domain_path = f"{root}/{domain}"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(domain_path, transform=transform)
        return dataset
    
    def get_data(self,root,domains,test_domain,train_split):
        train_domains = [domain for domain in domains if domain != test_domain]
        test_dataset = self.load_domain_data(root,test_domain)
        train_datasets = [self.load_domain_data(root,train_domain) for train_domain in train_domains]
        
        train_dataloaders = []
        valid_dataloaders = []
        
        for dataset in train_datasets:

            train_size = int(len(dataset) * train_split)
            val_size = len(dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

            train_dataloaders.append(train_dataloader)
            valid_dataloaders.append(valid_dataloader)

        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return  train_dataloaders, valid_dataloaders, test_dataloader
    

    def apply_randaugment(self,images, rand_augment):

        augmented_images = []
        for img in images:
            pil_img = transforms.ToPILImage()(img)
            augmented_img = rand_augment(pil_img)
            augmented_img = transforms.ToTensor()(augmented_img)
            augmented_images.append(augmented_img)
        return torch.stack(augmented_images)

    def train_clients(self,global_feature_extractor,global_classifer,train_loader,valid_loader,client_id):
        local_feature_extractor = copy.deepcopy(global_feature_extractor)
        local_classifer = copy.deepcopy(global_classifer)
        
        optimizer = torch.optim.SGD([{'params':local_feature_extractor.parameters(),'lr':self.lr},
                                            {'params':local_classifer.parameters(),'lr':self.lr}], momentum=0.9, weight_decay=5e-4)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.n_epochs, eta_min=0.0001)
        rand_augment = RandAugment(num_ops=2, magnitude=9)
        sum_loss = 0
        sum_acc = 0
        sum_valid_acc = 0
        features = []
        labels = []
        total_sample = 0
        for epoch in range(self.args.n_client_epochs):
            train_loss = 0
            train_correct = 0
            train_samples = 0
            valid_correct = 0
            valid_samples = 0
            total_sample = 0

            idx = 0
            local_feature_extractor.train()
            local_classifer.train()
            for idx,(data,target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                aug_data = self.apply_randaugment(data,rand_augment).to(self.device)

                optimizer.zero_grad()
                feature = local_feature_extractor(data)
                features.append(feature.detach())
                labels.append(torch.full((feature.size(0),), client_id, dtype=torch.long))

                output = local_classifer(feature)
                loss1 = torch.nn.CrossEntropyLoss()(output, target)
                loss1.backward(retain_graph=True)
                g_i = [
                    param.grad.view(-1)
                        for param in local_classifer.parameters() if param.grad is not None
                ]
                g_i = torch.cat(g_i)

                local_feature_extractor.zero_grad()
                local_classifer.zero_grad()

                output_global = global_classifer(local_feature_extractor(data))
                loss_global = torch.nn.CrossEntropyLoss()(output_global, target)
                loss_global.backward(retain_graph=True)
                g_global = [
                    param.grad.view(-1)
                        for param in global_classifer.parameters() if param.grad is not None
                ]
                g_global = torch.cat(g_global)

                local_feature_extractor.zero_grad()
                global_classifer.zero_grad()

                output_aug = local_classifer(local_feature_extractor(aug_data))
                loss2 = torch.nn.CrossEntropyLoss()(output_aug, target)
                loss2.backward(retain_graph=True)
                g_i_prime = [
                    param.grad.view(-1)
                    for param in local_classifer.parameters() if param.grad is not None
                ]
                g_i_prime = torch.cat(g_i_prime)

                local_feature_extractor.zero_grad()
                local_classifer.zero_grad()

                cosine_similarity_intra = F.cosine_similarity(g_i.unsqueeze(0), g_i_prime.unsqueeze(0))

                L_intra = 1 - cosine_similarity_intra.mean()

                cosine_similarity_inter = F.cosine_similarity(g_global.unsqueeze(0), g_i_prime.unsqueeze(0))

                L_inter = 1 - cosine_similarity_inter.mean()

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
            local_feature_extractor.eval()
            local_classifer.eval()
            with torch.no_grad():
                for idx,(data,target) in enumerate(valid_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = local_classifer(local_feature_extractor(data))
                    valid_correct += (output.argmax(1) == target).sum().item()
                    valid_samples += len(data)
                valid_acc = valid_correct / valid_samples
                sum_valid_acc += valid_acc
                print(
                    f"Client #{client_id} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Valid Acc: {valid_acc}",
                    end="\r",
                )
        scheduler.step()

        return local_feature_extractor, local_classifer, sum_loss/self.args.n_client_epochs, sum_acc/self.args.n_client_epochs, sum_valid_acc/self.args.n_client_epochs, torch.cat(features), torch.cat(labels), total_sample
    
    def train_server(self):
        train_losses = []
        train_acc = []
        for i in range(self.args.n_epochs):
            print(f"round {i+1}/{self.args.n_epochs}")

            self.clients_feature_extractor = []
            self.clients_classifer = []

            clients_losses = []
            client_accs = []
            client_valid_accs = []

            client_features = []
            client_labels = []

            client_samples = []
            total_sample = 0
            self.global_feature_extractor.train()
            self.global_classifer.train()

            for client_id in range(self.num_clients):
                client_feature_extractor, client_classifer, client_loss ,client_acc, client_valid_acc,client_feature, client_label, client_sample= self.train_clients(self.global_feature_extractor,
                                                                                             self.global_classifer,
                                                                                             self.train_loaders[client_id],
                                                                                             self.valid_loaders[client_id],
                                                                                             client_id)
                self.clients_feature_extractor.append(client_feature_extractor.state_dict())
                self.clients_classifer.append(client_classifer.state_dict())
                clients_losses.append(client_loss)
                client_accs.append(client_acc)
                client_valid_accs.append(client_valid_acc)
                client_features.append(client_feature)
                client_labels.append(client_label)
                
                client_samples.append(client_sample)

                total_sample += client_sample
            
            client_features = torch.cat(client_features)
            client_labels = torch.cat(client_labels)

            updated_weights_f = utils.static_avg(self.clients_feature_extractor,client_samples)
            updated_weights_c = utils.static_avg(self.clients_classifer,client_samples)

            self.global_feature_extractor.load_state_dict(updated_weights_f)
            self.global_classifer.load_state_dict(updated_weights_c)

            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            avg_acc = sum(client_accs) / len(client_accs)
            avg_valid_acc = sum(client_valid_accs) / len(client_valid_accs)
            train_acc.append(avg_acc)
            print(f"round {i+1}/{self.args.n_epochs} | Average Loss: {avg_loss} | Average train Acc: {avg_acc} | Average valid Acc: {avg_valid_acc}")

            fe_dataset = TensorDataset(client_features,client_labels)
            fe_dataloader = DataLoader(fe_dataset, batch_size=self.batch_size, shuffle=True)
            if i == 0:
                self.domain_classifier = domain_clf(512,self.num_clients,int(total_sample)).to(self.device)
            optimizer = torch.optim.SGD(self.domain_classifier.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

            fe_loss = 0
            fe_correct = 0
            idx = 0
            for idx,(data,target) in enumerate(fe_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(-1).unsqueeze(-1)
                optimizer.zero_grad()
                output = self.domain_classifier(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                fe_loss += loss.item()
                fe_correct += (output.argmax(1) == target).sum().item()
            
            print(f"round {i+1}/{self.args.n_epochs} | Classifier Loss: {fe_loss / (idx+1)} | Classifier Acc: {fe_correct/total_sample}")

            


    def test(self):
        self.global_feature_extractor.eval()
        self.global_classifer.eval()
        test_correct = 0
        test_sample = 0
        pbar = tqdm(self.test_loader, ncols=110)
        for idx, (data, target) in enumerate(pbar):

            data, target = data.to(self.device), target.to(self.device)

            # output = self.domain_classifier(self.global_feature_extractor(data).unsqueeze(-1).unsqueeze(-1))
            # output = output.squeeze(0)

            # new_updated_weights_f = utils.dynamic_avg(self.clients_feature_extractor,output)
            # new_updated_weights_c = utils.dynamic_avg(self.clients_classifer,output)
            # self.new_global_feature_extractor.load_state_dict(new_updated_weights_f)
            # self.new_global_classifer.load_state_dict(new_updated_weights_c)
            
            # output = self.new_global_classifer(self.new_global_feature_extractor(data))
            output = self.global_classifer(self.global_feature_extractor(data))
            test_correct += (output.argmax(1) == target).sum().item()
            test_sample += len(data)

        test_acc = test_correct / test_sample
        print(f"Test Accuracy: {test_acc}") 

        
    
if __name__ == "__main__":
    args = utils.arg_parser()
    fed = fed_dg(args)
    fed.train_server()
    fed.test()
    # fed_dg.test()
        
        

        

                
        

    
    





        
        

