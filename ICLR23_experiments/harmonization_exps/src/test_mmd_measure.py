import argparse
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import normalize

from src import dataloader as mydatasets, model as models
from src.average_meter import AverageMeter

def get_data(args, device, model_taunet, model_bnet, dataloader):
    X = []
    Y = []
    C = []
    G = []
    count = 0
    for images, labels, control, gattrs in dataloader:
        count += 1
        # Measure computed for a subset for Adult dataset
        if (args.dataset_name == 'Adult' or args.dataset_name == 'ACSIncome') and count > 70:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        gattrs = gattrs.to(device)
        _, _, latent, _ = model_taunet(images, gattrs.unsqueeze(-1))
        X.append(latent)
        Y.append(labels)
        C.append(control)
        G.append(gattrs)

    X = torch.cat(X)
    Y = torch.cat(Y)
    C = torch.cat(C)
    G = torch.cat(G)
    return X, Y, C, G


def compute_mmd_loss(args, device, mu, labels, logf):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float().to(device)
    unq_labels = torch.unique(labels)
    for i in unq_labels:
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    lap_kernel = lap_kernel * s_prod
    loss = -lap_kernel[c_diff == True].sum()
    loss += (len(unq_labels)-1)*lap_kernel[c_diff == False].sum()

    message = 'mmd_measure_loss {}'.format(100*loss) # Scaling loss by 100
    print(message)
    logf.write(message + '\n')

    
def run_mmd_measure(args, device, testset, equivar_model_name, invar_model_name, model_path, output_path, logf):
    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    dummy_x, _, _, _ = testset.__getitem__(0)
    input_dim = dummy_x.size(0)
    
    # LOADING THE MODELS FOR DIFFERENT CASES
        
    if (args.dataset_name == 'German' or args.dataset_name == 'Crimes'):
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
        
        model_bnet = None

        with torch.no_grad():        
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
            
    elif (args.dataset_name == 'Adult' or args.dataset_name == 'ACSIncome'):
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
        
        model_bnet = None

        with torch.no_grad():        
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
            
    else:
        raise NotImplementedError
