import torch.nn as nn
import torch.nn.functional as F
import torch
    

class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim):
        super(CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        output = self.fc(x)
        return output
    

class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        x = self.fc(x) / self.tmp 
        return x
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class OpenSet(nn.Module):
    """
    Prototype-based classifier
    Implements L2-normalization with a fully connected layer (no bias) 
    and an additional unknown category.
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(OpenSet, self).__init__()
        
        # Freeze the main classifier layer (fc) with normalized weights
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc.weight.requires_grad = False
        
        # Unknown classifier layer with learnable weights
        self.unknown = nn.Linear(in_dim, 1, bias=False)
        self.normalize_weights(self.unknown)
        
        # Temperature parameter for scaling the output
        self.temp = temp

        # Initialize weights of main classifier layer
        self.normalize_weights(self.fc)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = torch.cat([self.fc(x), self.unknown(x)], dim=1)
        x = x / self.temp
        return x
    
    def normalize_weights(self, layer):
        """L2-normalizes the weights of the specified layer."""
        w = layer.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        layer.weight.data = w.div(norm.expand_as(w))
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class ProtoNormCLS(nn.Module):
    """
    prototype-based classifier with auto normalization
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoNormCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp

    def forward(self, x):
        x = F.normalize(x)
        w = F.normalize(self.fc.weight)
        x = (x @ w.T) / self.tmp 
        return x


class Projection(nn.Module):
    """
    a projection head
    """
    def __init__(self, in_dim, hidden_mlp=2048, feat_dim=256):
        super(Projection, self).__init__()
        self.projection_head = nn.Sequential(
                               nn.Linear(in_dim, hidden_mlp),
                               nn.ReLU(inplace=True),
                               nn.Linear(hidden_mlp, feat_dim))
        self.output_dim = feat_dim

    def forward(self, x):
        return self.projection_head(x)
    

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x


