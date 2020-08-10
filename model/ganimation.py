import torch
from .base_model import BaseModel
from . import model_utils
import pickle

class GANimationModel(BaseModel):
    """docstring for GANimationModel"""
    def __init__(self):
        super(GANimationModel, self).__init__()
        self.name = "GANimation"

    def initialize(self, opt):
        super(GANimationModel, self).initialize(opt)

        self.net_gen = model_utils.define_splitG(self.opt.img_nc, self.opt.aus_nc, self.opt.ngf, signal_type = self.opt.control_signal_type,  init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)
        self.models_name.append('gen')
        
        if self.is_train:
            self.net_dis = model_utils.define_splitD(self.opt.img_nc, self.opt.aus_nc, self.opt.final_size, self.opt.ndf, signal_type = self.opt.control_signal_type, init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids)

            self.models_name.append('dis')

        self.net_dis_warp = model_utils.define_splitD(self.opt.img_nc, self.opt.aus_nc, self.opt.final_size, self.opt.ndf, signal_type = self.opt.control_signal_type, init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids, GAN_head = False, )

        self.models_name.append('dis_warp')

        if self.opt.load_epoch > 0:
            self.load_ckpt(self.opt.load_epoch)

    def setup(self):
        super(GANimationModel, self).setup()
        if self.is_train:
            # setup optimizer
            self.optim_gen = torch.optim.Adam(self.net_gen.parameters(),
                            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optims.append(self.optim_gen)
            self.optim_dis = torch.optim.Adam(self.net_dis.parameters(), 
                            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

            self.optim_dis_warp = torch.optim.Adam(self.net_dis_warp.parameters(), 
                            lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

            self.optims.append(self.optim_dis)
            self.optims.append(self.optim_dis_warp)
            # setup schedulers
            self.schedulers = [model_utils.get_scheduler(optim, self.opt) for optim in self.optims]

    def feed_batch(self, batch):
        self.src_img = batch['src_img'].to(self.device)
        if self.opt.control_signal_type in {'class', 'labelmap'}:
            totype = torch.LongTensor
        else:
            totype = torch.FloatTensor
        self.tar_aus = batch['tar_aus'].type(totype).to(self.device)
        if self.is_train or self.opt.control_signal_type in {'labelmap'}:
            self.src_aus = batch['src_aus'].type(totype).to(self.device)
            self.tar_img = batch['tar_img'].to(self.device)

    def feed_batch_dis(self, batch):
        self.src_img = batch['src_img'].to(self.device)

    def forward_dis(self):
        __, pred_real_aus = self.net_dis_warp(self.src_img)
        return pred_real_aus


    def forward(self, interp_coef = 1.):
        # generate fake image
        if self.opt.control_signal_type in {'labelmap', 'edgemap'}:
            self.img_warp_refine, self.img_warp, self.flow, self.cond_x = self.net_gen(self.src_img, [self.src_aus, self.tar_aus], interp_coef = interp_coef, coef=self.opt.coef_flow)
        else:
            self.img_warp_refine, self.img_warp, self.flow, self.cond_x = self.net_gen(self.src_img, self.tar_aus, interp_coef = interp_coef, coef=self.opt.coef_flow)
        self.fake_img = self.img_warp_refine

        # reconstruct real image
        if self.is_train:
          if self.opt.control_signal_type in {'labelmap', 'edgemap'}:
            self.rec_img_warp_refine, self.rec_img_warp, self.flow_rec, __ = self.net_gen(self.fake_img, [self.tar_aus, self.src_aus], interp_coef = interp_coef, coef=self.opt.coef_flow)
          else:
            self.rec_img_warp_refine, self.rec_img_warp, self.flow_rec, __ = self.net_gen(self.fake_img, self.src_aus, interp_coef = interp_coef, coef=self.opt.coef_flow)
          self.rec_real_img = self.rec_img_warp_refine

    def backward_dis(self):
        # real image
        pred_real, self.pred_real_aus = self.net_dis(self.src_img)
        self.loss_dis_real = self.criterionGAN(pred_real, True)
        if self.opt.control_signal_type in {'class', 'labelmap'}:
          self.loss_dis_real_aus = self.criterionCELoss(self.pred_real_aus, self.src_aus)
        else:
          self.loss_dis_real_aus = self.criterionMSE(self.pred_real_aus, self.src_aus)

        # fake image, detach to stop backward to generator
        pred_fake, _ = self.net_dis(self.fake_img.detach()) 
        self.loss_dis_fake = self.criterionGAN(pred_fake, False) 

        __, pred_real_aus = self.net_dis_warp(self.src_img)
        if self.opt.control_signal_type in {'class', 'labelmap'}:
          self.loss_dis_real_aus_warp = self.criterionCELoss(pred_real_aus, self.src_aus)
        else:
          self.loss_dis_real_aus_warp = self.criterionMSE(pred_real_aus, self.src_aus)

        # combine dis loss
        self.loss_dis =   self.opt.lambda_dis * (self.loss_dis_fake + self.loss_dis_real) \
                        + self.opt.lambda_aus * self.loss_dis_real_aus

        self.loss_dis_warp = self.loss_dis_real_aus_warp
        if self.opt.gan_type == 'wgan-gp':
            self.loss_dis_gp = self.gradient_penalty(self.src_img, self.fake_img)
            self.loss_dis = self.loss_dis + self.opt.lambda_wgan_gp * self.loss_dis_gp
        
        # backward discriminator loss
        self.loss_dis.backward()
        self.loss_dis_warp.backward()

    def backward_gen(self):
        # domain classification loss of refined warped image
        pred_fake, self.pred_fake_aus = self.net_dis(self.fake_img)
        self.loss_gen_GAN = self.criterionGAN(pred_fake, True)
        if self.opt.control_signal_type in {'class', 'labelmap'}:
          self.loss_gen_fake_aus = self.criterionCELoss(self.pred_fake_aus, self.tar_aus)
        else:
          self.loss_gen_fake_aus = self.criterionMSE(self.pred_fake_aus, self.tar_aus)

        # domain classification loss of warped image
        pred_fake_warp, self.pred_fake_aus_warp = self.net_dis_warp(self.img_warp)
        if self.opt.control_signal_type in {'class', 'labelmap'}:
          self.loss_gen_warp_aus = self.criterionCELoss(self.pred_fake_aus_warp, self.tar_aus)
        else:
          self.loss_gen_warp_aus = self.criterionMSE(self.pred_fake_aus_warp, self.tar_aus)

        # reconstruction loss
        self.loss_gen_rec = self.criterionL1(self.rec_real_img, self.src_img) + \
                            self.criterionL1(self.rec_img_warp, self.src_img)

        self.loss_ref = torch.mean(torch.abs(self.img_warp - self.fake_img))
        self.loss_ref_rec = torch.mean(torch.abs(self.rec_img_warp - self.rec_real_img))

        # combine and backward G loss
        self.loss_fake =   self.opt.lambda_dis * self.loss_gen_GAN \
                        + self.opt.lambda_aus * self.loss_gen_fake_aus 
        
        self.loss_rec = self.opt.lambda_rec * self.loss_gen_rec 

        self.loss_refine = self.opt.lambda_refine * (self.loss_ref + self.loss_ref_rec)

        self.loss_warp = self.opt.lambda_warp * self.loss_gen_warp_aus # 
                      
        self.loss_gen = self.loss_fake + self.loss_warp + self.loss_rec + self.loss_refine
        
        self.loss_gen.backward()

    def optimize_paras(self, train_gen):
        self.forward()
        # update discriminator
        self.set_requires_grad(self.net_dis, True)
        self.optim_dis.zero_grad()
        self.set_requires_grad(self.net_dis_warp, True)
        self.optim_dis_warp.zero_grad()

        self.backward_dis()
        self.optim_dis.step()
        self.optim_dis_warp.step()
        # update G if needed
        if train_gen:
            self.set_requires_grad(self.net_dis, False)
            self.optim_gen.zero_grad()
            self.backward_gen()
            self.optim_gen.step()

    def save_ckpt(self, epoch):
        # save the specific networks
        save_models_name = ['gen', 'dis', 'dis_warp']
        return super(GANimationModel, self).save_ckpt(epoch, save_models_name)

    def load_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['gen']
        if self.is_train:
            load_models_name.extend(['dis'])
        if 'emotionnet' not in self.opt.ckpt_dir:
            load_models_name.extend(['dis_warp'])
        return super(GANimationModel, self).load_ckpt(epoch, load_models_name)

    def clean_ckpt(self, epoch):
        # load the specific part of networks
        load_models_name = ['gen', 'dis', 'dis_warp']
        return super(GANimationModel, self).clean_ckpt(epoch, load_models_name)

    def get_latest_losses(self):
        get_losses_name = ['dis_fake', 'dis_real', 'dis_real_aus', 'dis_real_aus_warp', 'gen_rec'] \
                           + ['gen_GAN', 'ref', 'ref_rec', 'gen_fake_aus', 'gen_warp_aus']
        return super(GANimationModel, self).get_latest_losses(get_losses_name)

    def get_latest_visuals(self):
        visuals_name = ['src_img', 'tar_img', 'img_warp', 'img_warp_refine','mask']
        if self.is_train:
            visuals_name.extend(['rec_img_warp', 'rec_real_img', 'rec_mask'])
        return super(GANimationModel, self).get_latest_visuals(visuals_name)
