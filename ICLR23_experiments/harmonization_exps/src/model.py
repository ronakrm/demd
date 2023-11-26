import torch

from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F

class Adv(nn.Module):
    # New adversarial model for baseline which uses ReLU activation
    # The output would be logits
    def __init__(self, name='Adv', input_dim=64, output_dim=10, hidden_dim=64, hidden_layers=3):
        super(Adv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        self.output_dim = output_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(64))  # This is different from the previous adv
                layers.append(nn.ReLU())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.adv(x)
        if self.output_dim == 1:
            output = output.squeeze()
        return output
    

class OriginalBaselineAdv(nn.Module):
    def __init__(self, name='Adv', input_dim=10, output_dim=1, hidden_dim=64, hidden_layers=0):
        super(OriginalBaselineAdv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.Tanh())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.adv(x).squeeze()


class BaselineAdv(nn.Module):
    # New adversarial model for baseline which uses ReLU activation
    # The output would be logits
    def __init__(self, name='Adv', input_dim=10, output_dim=2, hidden_dim=64, hidden_layers=0):
        super(BaselineAdv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        self.output_dim = output_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.adv(x)
        if self.output_dim == 1:
            output = output.squeeze()
        return output



class FC_DetEnc(nn.Module):
    def __init__(self, input_dim=10, output_dim=10, hidden_dim=30, latent_dim=30):
        super(FC_DetEnc, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.name = 'Vib'
    
    def encode(self, x):
        in2 = self.relu(self.fc1(x))
        latent = self.relu(self.fc2(in2))
        return latent
    
    def predict(self, z):
        in4 = self.relu(self.fc3(z))
        out = self.fc4(in4)
        return out
        
    def forward(self, x):
        latent = self.encode(x)
        out = self.predict(latent)
        return out, latent


class BaselineEncDec(nn.Module):
    """
    There is one encoder, one decoder and one discriminator.
    This is replication of the baseline from tf from the dcmoyer/inv-rep repo uses ReLU activations.
    The final prediction output is logits.
    """
    def __init__(self, name='Main', input_dim=121, latent_dim=64, feature_dim=0):
        super(BaselineEncDec, self).__init__()
        
        self.name = name
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Encoder layers
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.latent_dim) 


        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim + self.feature_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # y-predictor
        self.fc5 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 2)  # One logit prediction for regression like task
        
        self.relu = nn.ReLU()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        mu = self.fc21(fc1)
        #logvar = self.fc22(fc1)
        return mu

    def decode(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4
    
    def predict(self, x):
        fc5 = self.relu(self.fc5(x))
        fc6 = self.fc6(fc5)
        return fc6

    def project_sphere(self, mu):
        mu_centered = (mu - torch.mean(mu, dim=1, keepdim=True))
        mu_norm = torch.norm(mu_centered, dim=1, keepdim=True)
        return mu_centered / mu_norm
    
    def forward(self, x, c):
        mu = self.encode(x)
        z = self.project_sphere(mu)
        recons = self.decode(mu)
        pred = self.predict(mu)
        return recons, pred, mu, None

class TauNetEncDec(nn.Module):
    """
    There is one encoder, one decoder and one discriminator.
    This is replication of the baseline from tf from the dcmoyer/inv-rep repo uses ReLU activations.
    The final prediction output is logits.
    """
    def __init__(self, name='Main', input_dim=121, latent_dim=64, feature_dim=0, const=0.01):
        super(TauNetEncDec, self).__init__()

        print('Using TauNetEncDec !!!')
        
        self.name = name
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.const=const
        
        # Encoder layers
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.latent_dim) 

        # TauNet layers
        #hidden_dim_t1 = 180
        hidden_dim_t2 = self.latent_dim #int((self.latent_dim * (self.latent_dim-1)) / 2) # N choose 2 for N=20
        self.fct1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fct2 = nn.Linear(self.latent_dim, hidden_dim_t2)

        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim + self.feature_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # y-predictor
        self.fc5 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 2)  # One logit prediction for regression like task
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        z1 = self.fc21(fc1)
        #logvar = self.fc22(fc1)
        return z1

    def decode(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4
    
    def predict(self, x):
        fc5 = self.relu(self.fc5(x))
        fc6 = self.fc6(fc5)
        return fc6

    def TauNet(self, z, gval):
        fct1 = self.relu(self.fct1(z))
        fct2 = self.fct2(fct1)

        fadj = gval + self.const*self.tanh(fct2)
        return fadj

    def encodeTauNet(self, x, gval):
        z = self.encode(x)
        z = self.project_sphere(z)
        g = self.TauNet(z, gval)
        return g

    def encodeProject(self, x):
        z = self.encode(x)
        z = self.project_sphere(z)
        return z

    def project_sphere(self, mu):
        mu_centered = (mu - torch.mean(mu, dim=1, keepdim=True))
        mu_norm = torch.norm(mu_centered, dim=1, keepdim=True)
        return mu_centered / mu_norm
    
    def forward(self, x, gval):
        z = self.encode(x)
        z = self.project_sphere(z)
        g = self.TauNet(z, gval)
        
        recons = self.decode(z)
        pred = self.predict(z)

        return recons, pred, g, None


class BNetEncDec(nn.Module):
    def __init__(self, device, name='Main', latent_dim=64, output_dim=64, feature_dim=0):
        super(BNetEncDec, self).__init__()

        print('Using BNetEncDec !!!')
        
        self.name = name
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.device = device
        hidden_dim = 64

        #Dim up
        group_dim = int((self.latent_dim*(self.latent_dim-1))/2)
        self.fcdimup = nn.Linear(self.latent_dim, group_dim)

        
        
        # BNet layers
        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim) 
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim) 

        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim + self.feature_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

        # y-predictor
        self.fc5 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 2)  # One logit prediction for regression like task
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def cayley_map(self, vec):
        N = vec.shape[0]
        A = torch.zeros((N, self.latent_dim, self.latent_dim)).to(self.device)
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        A[:, tril_indices[0], tril_indices[1]] = vec
        A = A.transpose(-1, -2) - A

        I = torch.eye(self.latent_dim).to(self.device)
        I = I.reshape((1, self.latent_dim, self.latent_dim))
        I = I.repeat(N, 1, 1)

        Q = I + A
        B = (I - A)
        norms = (torch.norm(B, p = 1, dim=(1,2)) * torch.norm(B, p = float('inf'), dim=(1,2)))
        Z = B.transpose(-1, -2)/ norms.view(N, 1, 1)
        for _ in range(10):
            Z = 0.25*Z@(13*I - B@Z@(15 * I - B@Z@(7*I - B @Z)))
        return Z@Q

    def encode(self, li_latent, gi_latent):
        gi_latent = self.fcdimup(gi_latent) # 128*30 -> 128*435
        gi_rot = self.cayley_map(gi_latent) # 128*435 -> 128*30*30

        # 128*30*30 X 128*30*1 = 128*30
        mis = (gi_rot.transpose(-1, -2) @ li_latent.unsqueeze(-1)).squeeze(-1)
        
        fc1 = self.relu(self.fc1(mis))
        fc2 = self.fc2(fc1)
        latent = (gi_rot @ fc2.unsqueeze(-1)).squeeze(-1)
        return latent

    def decode(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4
    
    def predict(self, x):
        fc5 = self.relu(self.fc5(x))
        fc6 = self.fc6(fc5)
        return fc6

    
    def forward(self, li_latent, gi_latent):
        latent = self.encode(li_latent, gi_latent)
        recons = self.decode(latent)
        pred = self.predict(latent)

        return recons, pred, latent
    

    
