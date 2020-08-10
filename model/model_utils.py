import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        # print("gpu_ids,", gpu_ids)
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


##############################################################################
# Classes
##############################################################################


class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan-gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
        if self.gan_type == 'wgan-gp':
            self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'gan':
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('GAN loss type [%s] is not found' % gan_type)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            target_tensor = target_is_real
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(dim, affine=True, track_running_stats=False),
                       nn.ReLU(True)]
        #if use_dropout:
        #    conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock2(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                       nn.ELU(True)]
        #if use_dropout:
        #    conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


##############################################################################
# Basic network model 
##############################################################################
def define_splitG(img_nc, aus_nc, ngf, signal_type='class', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net_img_au = Generator(img_nc, aus_nc, ngf, signal_type, repeat_num=6)
    return init_net(net_img_au, init_type, init_gain, gpu_ids)


def define_splitD(input_nc, aus_nc, image_size, ndf, signal_type='class', init_type='normal', init_gain=0.02, gpu_ids=[], GAN_head=True):
    net_dis_aus = Discriminator(input_nc, aus_nc, image_size, ndf, n_layers=6, GAN_head=GAN_head, control_signal_type = signal_type)
    return init_net(net_dis_aus, init_type, init_gain, gpu_ids)


class Generator(nn.Module):
    def __init__(self, img_nc, c_dim, conv_dim=64, signal_type=None, repeat_num=6, repeat_num2 = 4):
        assert(repeat_num >= 0)
        super(Generator, self).__init__()
        self.aus_nc = c_dim
        ### 128 scale
        ## Spontaneous motion module
        # encoder
        layers = []
        if signal_type in {'labelmap', 'edgemap'}:
            layers.append(nn.Conv2d(3 + 2 * c_dim, conv_dim, kernel_size=7, stride=2, padding=3, bias=False))
        else:
            layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResnetBlock(curr_dim))
        self.encoding = nn.Sequential(*layers)
        encoding_dim = curr_dim
        
        # decoder for motion prediction
        layers2 = []
        curr_dim = encoding_dim
        layers2.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers2.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
        layers2.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2
        layers2.append(nn.ConvTranspose2d(curr_dim, 2, kernel_size=6, stride=2, padding=2, bias=False))
        layers2.append(nn.Tanh())
        self.flow_pred = nn.Sequential(*layers2)
        

        ## Refinement module
        layers4 = []
        layers4.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=2, padding=3, bias=False))
        layers4.append(nn.ELU(inplace=True))
        curr_dim = conv_dim
        for i in range(repeat_num2):
            layers4.append(ResnetBlock2(curr_dim))
        layers4.append(nn.ConvTranspose2d(curr_dim, 3, kernel_size=6, stride=2, padding=2, bias=False))
        self.refine = nn.Sequential(*layers4)
        self.signal_type = signal_type
        if self.signal_type == 'class' or self.signal_type == 'labelmap':
            self.CONST_LOGITS = torch.arange(c_dim).unsqueeze(0)
            if self.signal_type == 'labelmap':
                self.CONST_LOGITS = self.CONST_LOGITS.unsqueeze(2).unsqueeze(3)



    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=0.1):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x


    def forward(self, img, c, interp_coef=1., coef=0.1):
        if self.signal_type == 'class':
            c = c.unsqueeze(1)
            c = (c == self.CONST_LOGITS.expand(c.size(0), self.CONST_LOGITS.size(1)).cuda()).float()
        elif self.signal_type == 'labelmap':
            assert isinstance(c, list), print('c must be a list of two iterms')
            cc = c[0].unsqueeze(1)
            #print(c.size(), self.CONST_LOGITS.size())
            logits = self.CONST_LOGITS.expand(cc.size(0), self.CONST_LOGITS.size(1), cc.size(2), cc.size(3)).cuda()
            c_src = (c[0].unsqueeze(1) == logits).float()
            c_tar = (c[1].unsqueeze(1) == logits).float()
            c = torch.cat([c_src, c_tar], dim=1)
        elif self.signal_type == 'edgemap':
            assert isinstance(c, list), print('c must be a list of two iterms')
            c = torch.cat([cc.unsqueeze(1) for cc in c], dim=1)
        if self.signal_type in {'class', 'au'}:
            c = c.unsqueeze(2).unsqueeze(3)
            c = c.expand(c.size(0), c.size(1), img.size(2), img.size(3))
        x_cond = torch.cat([img, c], dim=1)
        feat = self.encoding(x_cond)
        flow = self.flow_pred(feat) * interp_coef
        flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
        warp_x = self.warp(img, flow, coff=coef)
        refine_x = self.refine(warp_x)
        refine_warp_x = torch.clamp(refine_x, min=-1.0, max=1.0)
        return refine_warp_x, warp_x, flow, x_cond



"""
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)
"""



class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.InstanceNorm2d(inner_nc,  affine=True,)
        uprelu = nn.ReLU(True)
        upnorm = nn.InstanceNorm2d(outer_nc,  affine=True,)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)




class Discriminator(nn.Module):
    def __init__(self, input_nc, aus_nc, image_size=128, ndf=64, n_layers=6, GAN_head=True, control_signal_type='class'):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1

        self.GAN_head = GAN_head
        if self.GAN_head:
            self.dis_top = nn.Conv2d(ndf*4, 1, kernel_size=kw-1, stride=1, padding=padw, bias=False)
        self.control_signal_type = control_signal_type
        if control_signal_type in {'labelmap', 'edgemap'}:
          self.downlayer0 = nn.Sequential(*[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ])
          self.downlayer1 = nn.Sequential(*[
            nn.Conv2d(ndf, ndf*2, kernel_size=kw, stride=2, padding=padw, bias=True),
            nn.LeakyReLU(0.01, True)
        ])
          self.downlayer2 = nn.Sequential(*[
            nn.Conv2d(ndf*2, ndf*4, kernel_size=kw, stride=2, padding=padw, bias=True),
            nn.LeakyReLU(0.01, True)
        ])
          self.downlayer3 = nn.Sequential(*[
            nn.Conv2d(ndf*4, ndf*4, kernel_size=kw, stride=2, padding=padw, bias=True),
            nn.LeakyReLU(0.01, True)
        ])
          self.downlayer4 = nn.Sequential(*[
            nn.Conv2d(ndf*4, ndf*4, kernel_size=kw, stride=2, padding=padw, bias=True),
            nn.LeakyReLU(0.01, True)
        ])
          self.downlayer5 = nn.Sequential(*[
            nn.Conv2d(ndf*4, ndf*4, kernel_size=kw, stride=2, padding=padw, bias=True),
            nn.LeakyReLU(0.01, True)
        ])
          self.uplayer0 = nn.Sequential(nn.ConvTranspose2d(ndf*4,\
                           ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
             nn.LeakyReLU(0.01, True))
          self.uplayer1 = nn.Sequential(nn.ConvTranspose2d(ndf*8,\
                          ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
             nn.LeakyReLU(0.01, True))
          self.uplayer2 = nn.Sequential(nn.ConvTranspose2d(ndf*8,\
                           ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
             nn.LeakyReLU(0.01, True))
          self.uplayer3 = nn.Sequential(nn.ConvTranspose2d(ndf*8,\
                           ndf*4,kernel_size=4,stride=2,padding=1,bias=False),
             nn.LeakyReLU(0.01, True))
          self.uplayer4 = nn.Sequential(nn.ConvTranspose2d(ndf*6,\
                           ndf*3,kernel_size=4,stride=2,padding=1,bias=False),
             nn.LeakyReLU(0.01, True))
          self.aus_top = nn.ConvTranspose2d(ndf*4, aus_nc, kernel_size=4, stride=2,padding=1, bias=False) 
        else:
          use_bias = True
          sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
          ]

          cur_dim = ndf
          for n in range(1, n_layers):
            sequence += [nn.Conv2d(cur_dim, min(2 * cur_dim, 256),
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = min(2 * cur_dim, 256)

          self.model = nn.Sequential(*sequence)

          k_size = int(image_size / (2 ** n_layers))
          self.aus_top = nn.Conv2d(ndf*4, aus_nc, kernel_size=k_size, stride=1, bias=False)

    def forward(self, img):
        if self.control_signal_type in {'au', 'class'}:
            feat5 = self.model(img)
            pred_aus = self.aus_top(feat5)
        else:
            feat0 = self.downlayer0(img)
            feat1 = self.downlayer1(feat0)
            feat2 = self.downlayer2(feat1)
            feat3 = self.downlayer3(feat2)
            feat4 = self.downlayer4(feat3)
            feat5 = self.downlayer5(feat4)
            temp = self.uplayer0(feat5)
            temp = torch.cat([temp, feat4], dim=1)
            temp = self.uplayer1(temp)
            temp = torch.cat([temp, feat3], dim=1)
            temp = self.uplayer2(temp)
            temp = torch.cat([temp, feat2], dim=1)
            temp = self.uplayer3(temp)
            temp = torch.cat([temp, feat1], dim=1)
            temp = self.uplayer4(temp)
            temp = torch.cat([temp, feat0], dim=1)
            pred_aus = self.aus_top(temp)
        if self.GAN_head:
                pred_map = self.dis_top(feat5)
                return pred_map.squeeze(), pred_aus.squeeze()
        else:
                return None, pred_aus.squeeze()

