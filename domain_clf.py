from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F

#定义域分类器 其中输入维度为特征维度，num_clients为客户端数量，num_image为所有客户端的样本数，ratio为全连接层的缩放比例
#卷积层和全连接层的输入输出维度与论文里的设置一致
class domain_clf(nn.Module):
    def __init__(self,input_dim,num_clients,num_image,ratio=16):
        super(domain_clf,self).__init__()
        self.model = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim,num_image//num_clients,1),
            nn.Flatten(),
            nn.Linear(num_image//num_clients, num_image//num_clients//ratio),
            nn.ReLU(),
            nn.Linear(num_image//num_clients//ratio, num_clients),
        )
    
    def forward(self,x):
        return F.softmax(self.model(x),dim=-1)

        
    
    
    

    
        

    

    
        
        

