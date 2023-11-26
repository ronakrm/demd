import argparse
from src import dataloader as mydatasets, model as models
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt

from src.average_meter import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
sys.path.append('../..')


# In this we train the model on the actual dataset and not the latent space.
def get_prob(logits):
    logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
    logits = torch.exp(logits)
    prob = logits / logits.sum(dim=1).unsqueeze(1)
    return prob[:, 1]


def train_adv_epoch(args, device, epoch, adv, opt, dataloader, writer, model_taunet, model_bnet, tag='train'):
    # Function to train a single adversary
    loss_logger = AverageMeter()
    correct = 0
    total = 0
    pos = 0
    true_pos = 0
    start_time = time.time()
    if args.adv_use_weighted_loss:
        weights = dataloader.dataset.get_confound_weights().to(device)
    train = tag == 'train'
    if train:
        adv.train()
    else:
        adv.eval()
        
    y_true = np.array([])
    if args.dataset_name == 'Adult' or args.dataset_name == 'German' or args.dataset_name == 'Crimes':
        y_score = np.array([])
    elif args.dataset_name == 'ACSIncome':
        y_score = np.array([[], [], [], [], [], [], [], [], []]).transpose()
    else:
        raise NotImplementedError

    for idx, (x, _, c, g) in enumerate(dataloader):
        x = x.to(device)
        c = c.to(device)
        g = g.to(device)

        _, _, latent, _ = model_taunet(x, g.unsqueeze(-1))

        logits = adv(latent)
        
        # For computing the auc roc
        y_true = np.concatenate((y_true, c.cpu().numpy()))
        if args.dataset_name == 'Adult' or args.dataset_name == 'German' or args.dataset_name == 'Crimes':
            y_score = np.concatenate((y_score, get_prob(logits).detach().cpu().numpy()))
        elif args.dataset_name == 'ACSIncome':
            y_score_idx = torch.nn.Softmax(dim=1)(logits).detach().cpu().numpy()
            y_score = np.concatenate((y_score, y_score_idx))
        else:
            raise NotImplementedError
        
        pred = torch.argmax(logits, 1)
        correct += torch.sum(pred == c)
        total += x.size(0)
        pos += torch.sum(c)
        true_pos += torch.sum(c[pred == 1])
        if args.adv_use_weighted_loss:
            loss = F.cross_entropy(logits, c, weights)
        else:
            loss = F.cross_entropy(logits, c)
        loss_logger.update(loss.item())
        if idx % args.log_step == 0:
            #log_loss(epoch, args.adv_num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

    if args.dataset_name == 'Adult' or args.dataset_name == 'German' or args.dataset_name == 'Crimes':
        roc_auc = roc_auc_score(y_true, y_score)
    elif args.dataset_name == 'ACSIncome':
        roc_auc = -1. #not_computed
    else:
        raise NotImplementedError
    
    accuracy = correct.item() * 100.0 / total
    precision = true_pos * 100.0 / pos
    print(adv.name, tag, 'acc :', accuracy)
    print(adv.name, tag, 'precision :', precision.item())
    print(adv.name, tag, 'roc_auc :', roc_auc)
    writer.add_scalar(adv.name + '_loss/' + tag, loss_logger.avg, epoch)
    writer.add_scalar(adv.name + '_acc/' + tag, accuracy, epoch)
    writer.add_scalar(adv.name + '_precision/' + tag, precision.item(), epoch)
    writer.add_scalar(adv.name + '_roc_auc/' + tag, roc_auc, epoch)
    return accuracy, precision.item(), roc_auc

    

def train_adv(args, device, model_path, logf, adv, opt, trainloader, valloader, testloader, writer, model_taunet, model_bnet):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    for epoch in range(args.adv_num_epochs):
        print(' ')
        print('epoch', epoch)
        
        train_adv_epoch(args, device, epoch, adv, opt, trainloader, writer, model_taunet, model_bnet, tag='train')
        with torch.no_grad():
            if testloader is not None:
                train_adv_epoch(args, device, epoch, adv, opt, valloader, writer, model_taunet, model_bnet, tag='val')
                adv_test_acc, adv_precision, adv_roc_auc = train_adv_epoch(args, device, epoch, adv, opt, testloader, writer, model_taunet, model_bnet, tag='test')
            else:
                adv_test_acc, adv_precision, adv_roc_auc = train_adv_epoch(args, device, epoch, adv, opt, valloader, writer, model_taunet, model_bnet, tag='test')
        lr_scheduler.step()

    message = 'ADVERSARIAL test_acc{} precision{} roc_auc{}'.format(adv_test_acc, adv_precision, adv_roc_auc)
    print(message)
    logf.write(message + '\n')

    

def train_adversaries(args, device, model_path, logf, trainset, valset, testset, writer, model_taunet, model_bnet):

    input_dim = args.latent_dim
    if args.dataset_name == 'Adult' or args.dataset_name == 'German' or args.dataset_name == 'Crimes':
        output_dim = 2
    elif args.dataset_name == 'ACSIncome':
        output_dim = 9
    else:
        raise NotImplementedError
    
    hidden_dim = args.adv_hidden_dim

    if args.dataset_name == 'Adult' or args.dataset_name == 'German':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.adv_batch_size, shuffle=True, drop_last=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.adv_batch_size, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.adv_batch_size, shuffle=False, drop_last=True)
    elif args.dataset_name == 'ACSIncome' or args.dataset_name == 'Crimes':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.adv_batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.adv_batch_size, shuffle=False)
        testloader = None
    else:
        raise NotImplementedError
    
    adv_hidden_layers = args.adv_hidden_layers
    name = model_taunet.name + '_Adv'
    torch.manual_seed(args.seed)
    adv = models.Adv(name + str(adv_hidden_layers), input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                     hidden_layers=adv_hidden_layers).to(device)
    opt = optim.Adam(adv.parameters(), lr=args.adv_lr)
    train_adv(args, device, model_path, logf, adv, opt, trainloader, valloader, testloader, writer, model_taunet, model_bnet)


####################### Adversary after trained model #######################

def run_adversaries(args, device, model_path, logf, trainset, valset, testset, writer, equivar_model_name, invar_model_name):
    dummy_x, _, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)

    # Loading the equivar model        
    if args.dataset_name == 'German' or args.dataset_name == 'Crimes':
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
            
        model_bnet = None
        train_adversaries(args, device, model_path, logf, trainset, valset, testset, writer, model_taunet, model_bnet)
            
    elif args.dataset_name == 'Adult' or args.dataset_name == 'ACSIncome':
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()

        model_bnet = None
        train_adversaries(args, device, model_path, logf, trainset, valset, testset, writer, model_taunet, model_bnet)
            
    else:
        raise NotImplementedError

