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

from src.test_adversaries import train_adv_epoch

from demdFunc import dEMD, OBJ
from demdLoss import dEMDLossFunc
from demdLayer import *

def get_equiv_loss(args, device, g_lt, g):

    # Constant difference
    g_lt_freeze = g_lt.detach().clone()    
    g_lt_diff = g_lt.unsqueeze(1) - g_lt_freeze.unsqueeze(0)

    g_diff = (g.unsqueeze(1) - g.unsqueeze(0)).unsqueeze(-1)
    normal_ = torch.normal(0, 0.001, size=g_lt_diff.shape).to(device)
    delta = g_diff + normal_    
    temp = (args.delta_lambda * delta + g_lt_diff)
    loss = (temp * temp).sum(-1).mean()

    return loss


def mmd_lap_loss(args, device, mu, labels):
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
    
    return loss

def pairwise_loss(args, device, mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.pow(2).sum(-1)
    
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
    diff = diff * s_prod
    loss = diff[c_diff == True].sum()
    loss += -(len(unq_labels)-1)*diff[c_diff == False].sum()
    
    return loss


def train_disc(args, device, epoch, adv, opt, trainloader, writer, equivar_model):
    assert equivar_model is not None
    equivar_model.eval()
    train_adv_epoch(args, device, epoch, adv, opt, trainloader, writer, equivar_model, None, tag='train')
    print('Disc epoch!')


def train_disentangler(device, epoch, dis, opt, trainloader, writer, equivar_model):
    equivar_model.eval()
    for idx, (x, y, c, g) in enumerate(trainloader):
        x = x.to(device)
        _, _, e1, e2 = equivar_model(x)
        e1 = e1.detach()
        e2 = e2.detach()
        e1_pred, e2_pred = dis(e1, e2)
        loss = F.mse_loss(e1_pred, e1) + F.mse_loss(e2_pred, e2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('Disentangler epoch!')


def equivar_epoch(args, device, epoch, model, opt, dataloader, writer, tag='train', disc=None, disc_opt=None, demd_loss=None):
    loss_logger = AverageMeter()
    recons_loss_logger = AverageMeter()
    pred_loss_logger = AverageMeter()
    equiv_loss_logger = AverageMeter()
    mu_logger = AverageMeter()
    sigma_logger = AverageMeter()
    prior_loss_logger = AverageMeter()
    train = tag == 'train'
    if train:
        if args.equiv_type == 'cai':
            train_disc(args, device, epoch, disc, disc_opt, dataloader, writer, model)
            disc.eval()
        model.train()
    else:
        model.eval()

    total_steps = len(dataloader.dataset)//args.batch_size
    y_correct = 0
    y_total = 0
    y_true_pos = 0
    y_pos = 0
    start_time = time.time()

    for idx, (x, y, c, g) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        g = g.to(device)

        recons, pred_logits, g_lt, _ = model(x, g.unsqueeze(1))

        if args.equiv_type == 'mmd_lap':
            equiv_loss = mmd_lap_loss(args, device, g_lt, c)
        elif args.equiv_type == 'cai':
            if train:
                logits = disc(g_lt)
                equiv_loss = -1 * F.cross_entropy(logits, c)
            else:
                equiv_loss = torch.tensor(0).to(device)
        elif args.equiv_type == 'none':
            equiv_loss = torch.tensor(0).to(device)
        elif args.equiv_type == 'demd':
            equiv_loss = 0.
            for feat_id in range(g_lt.shape[-1]):
                equiv_loss += demd_loss(g_lt[:, feat_id], c)
            
        else:
            raise NotImplementedError

        if recons is not None:
            recons_loss = F.mse_loss(recons, x)
        else:
            recons_loss = torch.tensor(0).to(device)
                
        if pred_logits is not None:
            pred_loss = F.cross_entropy(pred_logits, y)
        else:
            pred_loss = torch.tensor(0).to(device)
                
        loss = args.recon_lambda * recons_loss + pred_loss + args.equiv_lambda *  equiv_loss
                
        if args.add_prior:
            loss += args.beta * prior_loss
            prior_loss_logger.update(prior_loss.item())

        mu_logger.update(g_lt.norm(dim=-1).mean())
        
        # Log the losses
        loss_logger.update(loss.item())
        recons_loss_logger.update(recons_loss.item())
        if pred_logits is not None:
            pred_loss_logger.update(pred_loss.item())
        equiv_loss_logger.update(equiv_loss.item())

        pred = torch.argmax(pred_logits, 1)
        y_correct += torch.sum(pred == y)
        y_total += x.size(0)
        y_pos += torch.sum(y)
        y_true_pos += torch.sum(y[pred == 1])

        if idx % args.log_step == 0:
            start_time = time.time()
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model_name = 'equivar_'
    accuracy = y_correct * 100.0 / y_total
    precision = y_true_pos * 100.0 / y_pos
    recons_loss_avg = recons_loss_logger.avg
    print(tag, 'accuracy:', accuracy.item(), 'recons_loss:', recons_loss_avg)
    
    writer.add_scalar(model_name + 'acc/' + tag, accuracy, epoch)
    writer.add_scalar(model_name + 'recons_loss/' + tag, recons_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'pred_loss/' + tag, pred_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'equiv_loss/' + tag, equiv_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'mu/' + tag, mu_logger.avg, epoch)
    writer.add_scalar(model_name + 'loss/' + tag, loss_logger.avg, epoch)
    return accuracy, recons_loss_avg


def train_equivar(args, device, model_path,  logf, model, opt, trainloader, valloader, testloader, writer):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    best_val_acc = 0    
    if args.equiv_type == 'cai':
        if args.dataset_name == 'German' or args.dataset_name == 'Adult' or args.dataset_name == 'Crimes':
            disc = models.Adv('Disc', input_dim=args.latent_dim, output_dim=2,
                              hidden_dim=args.adv_hidden_dim, hidden_layers=3).to(device)
        elif args.dataset_name == 'ACSIncome':
            disc = models.Adv('Disc', input_dim=args.latent_dim, output_dim=9,
                              hidden_dim=args.adv_hidden_dim, hidden_layers=3).to(device)
        else:
            raise NotImplementedError
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)

    if args.equiv_type == 'demd':
        demd_loss = DEMDLayer(discretization=args.demd_nbins).to(device)

    for epoch in range(1, args.num_epochs + 1):
        if args.equiv_type == 'cai':
            equivar_epoch(args, device, epoch, model, opt, trainloader, writer, tag='train', disc=disc, disc_opt=disc_opt)
            val_acc, val_recons = equivar_epoch(args, device, epoch, model, opt, valloader, writer, tag='val')
            if testloader is None:
                test_acc, test_recons = None, None
            else:
                test_acc, test_recons = equivar_epoch(args, device, epoch, model, opt, testloader, writer, tag='test')
        elif args.equiv_type == 'demd':
            equivar_epoch(args, device, epoch, model, opt, trainloader, writer, tag='train', demd_loss=demd_loss)
            val_acc, val_recons = equivar_epoch(args, device, epoch, model, opt, valloader, writer, tag='val', demd_loss=demd_loss)
            if testloader is None:
                test_acc, test_recons = None, None
            else:
                test_acc, test_recons = equivar_epoch(args, device, epoch, model, opt, testloader, writer, tag='test', demd_loss=demd_loss)            
        else:
            equivar_epoch(args, device, epoch, model, opt, trainloader, writer, tag='train')
            val_acc, val_recons = equivar_epoch(args, device, epoch, model, opt, valloader, writer, tag='val')
            if testloader is None:
                test_acc, test_recons = None, None
            else:
                test_acc, test_recons = equivar_epoch(args, device, epoch, model, opt, testloader, writer, tag='test')
            
        if val_acc > best_val_acc:
            name = 'Equivar_best_val_acc'
            model.name = name
            path = os.path.join(model_path, name + '.pth')
            torch.save(model.state_dict(), path)
            best_val_acc = val_acc
            message = 'Best val_acc{} val_recons{}\n test_acc{} test_recons{}\n Saving model{}\n'.format(
                best_val_acc, val_recons, test_acc, test_recons, path)
            print(message)
            logf.write(message + '\n')
        if epoch % args.save_step == 0:
            name = 'Equivar_ckpt_' + str(epoch)
            path = os.path.join(model_path, name + '.pth')
            model.name = name
            torch.save(model.state_dict(), path)
        lr_scheduler.step()
    name = 'Equivar'
    model.name = name
    path = os.path.join(model_path, name + '.pth')
    torch.save(model.state_dict(), path)


def run_equivariance(args, device, model_path, logf, trainset, valset, testset, writer):
    if args.dataset_name == 'German' and (args.equiv_type == 'cai'):
        drop_last = True
    else:
        drop_last = False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              drop_last=drop_last)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    if testset is None:
        testloader = None
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    dummy_x, _, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)

    if args.dataset_name == 'German' or args.dataset_name == 'Crimes':
        model = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
    elif args.dataset_name == 'Adult' or args.dataset_name == 'ACSIncome':
        model = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
    else:
        raise NotImplementedError
            
    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_equivar(args, device, model_path, logf, model, opt, trainloader, valloader, testloader, writer)
