from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class getDataset_train(Dataset):
    def __init__(self,root,domain):
        self.path = os.path.join(root, domain)
        self.samples = []
        self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform_aug = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        for laber_idx,file_name in enumerate(os.listdir(self.path)):
            file_path = os.path.join(self.path, file_name)
            if os.path.isdir(file_path):
                for img_name in os.listdir(file_path):
                    img_path = os.path.join(file_path, img_name)
                    self.samples.append((img_path, laber_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_orig = self.transform(image)
        image_aug = self.transform_aug(image)
        return image_orig, image_aug, label
    
class getDataset_test(Dataset):
    def __init__(self,root,domain):
        self.path = os.path.join(root, domain)
        self.samples = []
        self.transform = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),  # 对应 Inter.LINEAR
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
        for laber_idx,file_name in enumerate(os.listdir(self.path)):
            file_path = os.path.join(self.path, file_name)
            if os.path.isdir(file_path):
                for img_name in os.listdir(file_path):
                    img_path = os.path.join(file_path, img_name)
                    self.samples.append((img_path, laber_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        return image,label