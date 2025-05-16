from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms

class PACSDataset(Dataset):
    def __init__(self,txt_file):
        self.base_dir = '/newdata3/wsj/PACS/pacs_data/pacs_data'
        self.samples = []
        self.transform = transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with open(txt_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.base_dir, path)
                self.samples.append((full_path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label-1
    

def get_pacs_data(test_domain):
    train_loaders = []
    valid_loaders = []
    domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    domains.remove(test_domain)
    for i in range(3):
        train_path = os.path.join( '/newdata3/wsj/PACS/pacs_label',domains[i]+"_train_kfold.txt")
        train_dataset = PACSDataset(train_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        train_loaders.append(train_loader)

        validation_path = os.path.join( '/newdata3/wsj/PACS/pacs_label',domains[i]+"_crossval_kfold.txt")
        valid_dataset = PACSDataset(validation_path)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)
        valid_loaders.append(valid_loader)
    test_path = os.path.join( '/newdata3/wsj/PACS/pacs_label',test_domain+"_test_kfold.txt")
    test_dataset = PACSDataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loaders, valid_loaders, test_loader