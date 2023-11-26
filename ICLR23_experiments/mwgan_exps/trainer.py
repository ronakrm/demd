import os
import time
import datetime
import itertools
import copy

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import ResEncoder, ResDecoder
from model import Discriminator

from demd import DEMDLayer, dEMD

class Trainer(object):
    """
    Trainer for training and testing MWGAN.
    """
    def __init__(self, src_loader, tgt_loaders, opts):
        self.src_loader = src_loader
        self.tgt_loaders = tgt_loaders
        self.opts = opts
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # criterion function
        self.criterionIdt = torch.nn.L1Loss()

        self.interType = opts.regType
        if opts.regType == 'demd':
            self.reg = DEMDLayer(discretization=opts.demd_nbins).to(self.device)
        elif opts.regType == '3M':
            self.reg = torch.nn.Identity()
        elif opts.regType == 'none':
            self.reg = torch.nn.Identity()

        # build models
        self.build_model()

        # use logger if allowed
        if opts.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.logger = SummaryWriter(opts.log_dir)




    def build_model(self):
        """
        Build models and initialize optimizers.
        """
        # build shared encoder
        self.E = ResEncoder(self.opts.g_conv_dim, self.opts.g_repeat_num).to(self.device)

        # build decoders(also known as generator)
        self.Gs = []
        for i in range(self.opts.num_domains - 1):
            G_i = ResDecoder(self.opts.g_conv_dim, self.opts.g_repeat_num).to(self.device)
            self.Gs.append(G_i)

        # build discriminator( combined with the auxiliary classifier )
        self.D = Discriminator(self.opts.image_size, self.opts.d_conv_dim, self.opts.num_domains - 1, self.opts.d_repeat_num).to(self.device)


        # build optimizers
        param_list = [self.E.parameters()] + [G.parameters() for G in self.Gs]
        self.g_optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                            self.opts.g_lr, [self.opts.beta1, self.opts.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.opts.d_lr, [self.opts.beta1, self.opts.beta2])


    def restore_model(self, resume_iters):
        """
        Restore the trained generators and discriminator.
        """
        print('Loading the trained models from step {}...'.format(resume_iters))

        E_path = os.path.join(self.opts.checkpoint_dir, '{}-E.ckpt'.format(resume_iters))
        self.E.load_state_dict(torch.load(E_path, map_location=lambda storage, loc: storage))

        for i in range(self.opts.num_domains - 1):
            G_i_path = os.path.join(self.opts.checkpoint_dir, '{}-G{}.ckpt'.format(resume_iters, i+1))
            self.Gs[i].load_state_dict(torch.load(G_i_path, map_location=lambda storage, loc: storage))

        D_path = os.path.join(self.opts.checkpoint_dir, '{}-D.ckpt'.format(resume_iters))
        if os.path.exists(D_path):
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def gradient_penalty(self, y, x, Lf):
        """
        Compute gradient penalty.
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))

        ZERO = torch.zeros_like(dydx_l2norm).to(self.device)
        penalty = torch.max(dydx_l2norm - Lf, ZERO)
        return torch.mean(penalty) ** 2

    def gradient_penalty_nomean(self, y, x, Lf):
        """
        Compute gradient penalty.
        """
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))

        return dydx_l2norm


    def classification_loss(self, logit, target, type='BCE'):
        """
        Compute classification loss.
        """
        if type == 'BCE':
            return F.binary_cross_entropy_with_logits(logit, target)
        elif type == 'LS':
            return F.mse_loss(logit, target)
        else:
            assert False, '[*] classification loss not implemented.'

    def disc_loss(self, x_src, x_tgts, embedder=None, disc=None, gens=None):

        if embedder is None:
            embedder = self.E
        if disc is None:
            disc = self.D
        if gens is None:
            gens = self.Gs

        embedding = embedder(x_src).detach()
        x_fake_list = []
        x_src_list = []
        d_loss_cls = 0
        d_loss_fake = 0
        attr_list = []
        for idx in range(len(gens)):
            x_fake_i = gens[idx](embedding).detach()
            x_fake_list.append(x_fake_i)
            x_src_list.append(x_src)

            out_fake_i, out_cls_fake_i = disc(x_fake_i)
            _, out_cls_real_i = disc(x_tgts[idx])

             # domain classification loss
            d_loss_cls_i = self.classification_loss(out_cls_real_i[:, idx], self.label_pos, type=self.opts.cls_loss) \
                            + self.classification_loss(out_cls_fake_i[:, idx], self.label_neg, type=self.opts.cls_loss)
            d_loss_cls += d_loss_cls_i

            if self.interType == 'demd' or self.interType == 'splitmean' or self.interType == 'splitmeandemd' or self.interType == 'demdmean':
                    attr_idx = torch.full((self.opts.batch_size,1),idx)
                    attr_list.append(torch.squeeze(attr_idx))

            # part of adversarial loss
            d_loss_fake += torch.mean(out_fake_i)

        out_src, out_cls_src = disc(x_src)
        # adversarial loss
        d_loss_adv = torch.mean(out_src) - d_loss_fake / (self.opts.num_domains - 1)

        d_loss_reg = torch.tensor(0.).to(self.device)#.requires_grad_(True)
        d_demd_reg = torch.tensor(0.).to(self.device)#.requires_grad_(True)
        # inter-domain gradient penalty
        if self.interType == 'demd':
            # push incorrect gen-disc pipelines to be close to each other for other
            labels = torch.hstack(attr_list).to(self.device)
            dists = []
            for j in range(len(gens)):
                x_hat_j = x_fake_list[j].data.requires_grad_(True)
                out_hat_j, _ = disc(x_hat_j)
                dists.append(self.gradient_penalty_nomean(out_hat_j, x_hat_j, self.opts.Lf))

            values = torch.hstack(dists)
            labels = torch.hstack(attr_list).to(self.device)
            d_demd_reg = self.reg(values, labels)

        elif self.interType == '3M':
            # compute loss for gradient penalty.
            x_fake_cat = torch.cat(x_fake_list)
            x_src_cat = torch.cat(x_src_list)
            alpha = torch.rand(x_src_cat.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_src_cat.data + (1 - alpha) * x_fake_cat.data).requires_grad_(True)
            out_hat, _ = disc(x_hat)
            d_loss_reg = self.gradient_penalty(out_hat, x_hat, self.opts.Lf)

        # total loss for update discriminator
        d_loss = -1 * d_loss_adv + self.opts.lambda_cls * d_loss_cls + self.opts.lambda_reg * d_loss_reg + self.opts.demd_reg * d_demd_reg

         
        return d_loss, d_loss_adv, d_loss_cls, d_loss_reg, d_demd_reg

    def gen_loss(self, x_src, x_tgts, embedder=None, disc=None, gens=None):

        if embedder is None:
            embedder = self.E
        if disc is None:
            disc = self.D
        if gens is None:
            gens = self.Gs

        embedding = embedder(x_src)

        g_loss_info = 0
        g_loss_adv = 0
        g_loss_idt = 0

        for idx in range(len(gens)):
            x_fake_i = gens[idx](embedding)

            if self.opts.lambda_idt > 0:
                x_fake_i_idt = gens[idx](embedder(x_tgts[idx]))
                g_loss_idt += self.criterionIdt(x_fake_i_idt, x_tgts[idx])

            out_fake_i, out_cls_fake_i = disc(x_fake_i)

            # mutual information maximization
            g_loss_info_i = F.binary_cross_entropy_with_logits(out_cls_fake_i[:, idx], self.label_pos)
            g_loss_info += g_loss_info_i

            # adversarial loss
            g_loss_adv -= torch.mean(out_fake_i) # opposed sign

        # total loss for update generator
        g_loss = g_loss_adv / (self.opts.num_domains - 1) + self.opts.lambda_info * g_loss_info + self.opts.lambda_idt * g_loss_idt

        return g_loss, g_loss_adv, g_loss_info, g_loss_idt
            

    def train(self):
        """
        Train MWGAN
        """
        src_iter = iter(self.src_loader)
        tgt_iters = []
        for loader in self.tgt_loaders:
            tgt_iters.append(iter(loader))

        # fixed data for evaluating: generate samples.
        x_src_fixed, _ = next(src_iter)
        x_src_fixed = x_src_fixed.to(self.device)

        # label
        self.label_pos = torch.FloatTensor([1] * self.opts.batch_size).to(self.device)
        self.label_neg = torch.FloatTensor([0] * self.opts.batch_size).to(self.device)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.opts.resume_iters:
            start_iters = self.opts.resume_iters
            self.restore_model(self.opts.resume_iters)

        disc_grad_eta = []
        gen_grad_eta = []

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.opts.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch images from domains
            try:
                x_src, _ = next(src_iter)
            except:
                src_iter = iter(self.src_loader)
                x_src, _ = next(src_iter)

            x_tgts = []
            for tgt_idx in range(len(tgt_iters)):
                try:
                    x_tgt_i, _ = next(tgt_iters[tgt_idx])
                    x_tgts.append(x_tgt_i)
                except:
                    tgt_iters[tgt_idx] = iter(self.tgt_loaders[tgt_idx])
                    x_tgt_i, _ = next(tgt_iters[tgt_idx])
                    x_tgts.append(x_tgt_i)

            x_src = x_src.to(self.device)
            for tgt_idx in range(len(x_tgts)):
                x_tgts[tgt_idx] = x_tgts[tgt_idx].to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #


            d_loss, d_loss_adv, d_loss_cls, d_loss_reg, d_demd_reg = self.disc_loss(x_src, x_tgts)

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_tot'] = d_loss.item()
            loss['D/loss_emd'] = d_demd_reg.item()
            loss['D/loss_adv'] = d_loss_adv.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_reg'] = d_loss_reg.item()

            # =================================================================================== #
            #                               3. Train the generators                               #
            # =================================================================================== #

            if (i + 1) % self.opts.n_critic == 0:
                
                g_loss, g_loss_adv, g_loss_info, g_loss_idt = self.gen_loss(x_src, x_tgts)

                self.reset_grad()
                g_loss.backward()                
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_tot'] = g_loss.item()
                loss['G/loss_adv'] = g_loss_adv.item()
                loss['G/loss_cls'] = g_loss_info.item()
                if self.opts.lambda_idt > 0:
                    loss['G/loss_idt'] = g_loss_idt.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.opts.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.opts.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.add_scalar(tag, value, i + 1)

            # save translated samples.
            if (i + 1) % self.opts.sample_step == 0:
                # model.eval()
                for idx in range(len(self.Gs)):
                    self.Gs[idx].eval()

                with torch.no_grad():
                    x_fake_list = [x_src_fixed]
                    embedding_fixed = self.E(x_src_fixed)

                    for idx in range(len(self.Gs)):
                        x_fake_i_fixed = self.Gs[idx](embedding_fixed)
                        x_fake_list.append(x_fake_i_fixed)

                    # produce composite results (note that the code here is not flexible!).
                    composite_translation = False
                    if composite_translation and self.opts.num_domains == 5:
                        x_fake_12 = self.Gs[1](self.E(x_fake_list[1]))
                        x_fake_13 = self.Gs[2](self.E(x_fake_list[1]))
                        x_fake_123 = self.Gs[2](self.E(x_fake_12))
                        x_fake_list.extend([x_fake_12, x_fake_13, x_fake_123])

                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.opts.sample_dir, '{:0>6d}-images.jpg'.format(i + 1))
                    save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0, normalize=True)
                    print('Saved samples into {}...'.format(sample_path))

                # return model.train()
                for idx in range(len(self.Gs)):
                    self.Gs[idx].train()

            # save model checkpoints.
            if (i + 1) % self.opts.model_save_step == 0:
                E_path = os.path.join(self.opts.checkpoint_dir, '{}-E.ckpt'.format(i + 1))
                torch.save(self.E.state_dict(), E_path)

                D_path = os.path.join(self.opts.checkpoint_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.D.state_dict(), D_path)

                for idx in range(len(self.Gs)):
                    G_i_path = os.path.join(self.opts.checkpoint_dir, '{}-G{}.ckpt'.format(i + 1, idx+1))
                    torch.save(self.Gs[idx].state_dict(), G_i_path)

                print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))

    def test(self):
        """
        Translate images using MWGAN.
        """
        # Load the trained generator.
        self.restore_model(self.opts.test_iters)

        # Set data loader.
        src_loader = self.src_loader

        # model.eval()
        for idx in range(len(self.Gs)):
            self.Gs[idx].eval()

        with torch.no_grad():
            for i, (x_src, c_org) in enumerate(src_loader):

                # Prepare input images
                x_src = x_src.to(self.device)

                # Translate images.
                x_fake_list = [x_src]
                embedding = self.E(x_src)

                for idx in range(len(self.Gs)):
                    x_fake_i = self.Gs[idx](embedding)
                    x_fake_list.append(x_fake_i)

                # produce composite results (note that the code here is not flexible!).
                composite_translation = False
                if composite_translation and self.opts.num_domains == 5:
                    x_fake_12 = self.Gs[1](self.E(x_fake_list[1]))
                    x_fake_13 = self.Gs[2](self.E(x_fake_list[1]))
                    x_fake_123 = self.Gs[2](self.E(x_fake_12))
                    x_fake_list.extend([x_fake_12, x_fake_13, x_fake_123])

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.opts.result_dir, '{:0>5d}-images.jpg'.format(i+1))
                save_image(x_concat.data.cpu(), result_path, nrow=1, padding=0, normalize=True)
                print('Saved real and fake images into {}...'.format(result_path))


    def test_separate(self):
        """
        Translate images using MWGAN.
        """
        # Load the trained generator.
        self.restore_model(self.opts.test_iters)

        # Set data loader.
        src_loader = self.src_loader

        # model.eval()
        for idx in range(len(self.Gs)):
            self.Gs[idx].eval()

        with torch.no_grad():

            sample_cnt = 0
            for i, (x_src, c_org) in enumerate(src_loader):

                # Prepare input images
                x_src = x_src.to(self.device)

                # Translate images.
                x_fake_list = [x_src]
                embedding = self.E(x_src)

                for idx in range(len(self.Gs)):
                    x_fake_i = self.Gs[idx](embedding)

                    batchcnt = sample_cnt
                    for s in range(x_fake_i.shape[0]):

                        # Save the translated images.
                        result_path = os.path.join(self.opts.result_dir, 'domain_{:d}/{:0>5d}-images.jpg'.format(idx, batchcnt))
                        save_image(x_fake_i[s,:].cpu(), result_path, nrow=1, padding=0, normalize=True)
                        print('Saved real and fake images into {}...'.format(result_path))

                        batchcnt += 1

                    sample_cnt += x_fake_i.shape[0]