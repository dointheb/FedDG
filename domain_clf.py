from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F

class domain_clf(Dataset):
    def __init__(self,input_dim,num_clients,num_image,ratio=16):
        super(domain_clf,self).__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim,num_image//num_clients,1),
            nn.Linear(num_image//num_clients, num_image//num_clients//ratio),
            nn.ReLU(),
            nn.Linear(num_image//num_clients//ratio, num_clients),
        )
    
    def forward(self,x):
        return F.softmax(self.model(x),dim=-1)

        
    
    
    

    
        

    

    
        
        

