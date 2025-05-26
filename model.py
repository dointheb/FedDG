import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1):
        super(MixStyle, self).__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):
        if not self.training or torch.rand(1) > self.p:
            return x

        B = x.size(0)
        feat = x.view(B, x.size(1), -1)  # [B, C, HW]
        mu = feat.mean(dim=2, keepdim=True)
        sigma = feat.std(dim=2, keepdim=True) + 1e-6  # 加上epsilon，防止除0

        # Shuffle
        perm = torch.randperm(B)
        mu2, sigma2 = mu[perm], sigma[perm]

        # Sample mixing coefficient from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu2 * (1 - lam)
        sigma_mix = sigma * lam + sigma2 * (1 - lam)

        feat_norm = (feat - mu) / sigma
        feat_mix = feat_norm * sigma_mix + mu_mix
        return feat_mix.view_as(x)
    
class ResNetWithMixStyle(ResNet):
    def __init__(self, mix_layers=[], mix_p=0.5, mix_alpha=0.1, **kwargs):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)

        self.mixstyle = MixStyle(p=mix_p, alpha=mix_alpha)
        self.mix_layers = mix_layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.mix_layers:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if 'layer2' in self.mix_layers:
            x = self.mixstyle(x)
        x = self.layer3(x)
        if 'layer3' in self.mix_layers:
            x = self.mixstyle(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
class OfficeHomeDGClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def get_grads(self):
        return torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])

    def set_grads(self, new_grads):
        start = 0
        for p in self.parameters():
            if p.grad is None:
                continue
            numel = p.numel()
            p.grad.copy_(new_grads[start:start+numel].view_as(p))
            start += numel