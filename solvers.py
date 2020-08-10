from data import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy
import time
import os
import torch
import numpy as np
from PIL import Image
import random
import pickle
import torchvision.transforms as transforms
import glob
import cv2
import skimage.transform as sktransform
import cvbase as cvb

INTER_TYPE = cv2.INTER_LINEAR #cv2.INTER_NEAREST #cv2.INTER_LINEAR INTER_CUBIC
FIX_TAR = False
FLOW_INTER_TYPE = cv2.INTER_NEAREST
REPEAT_NUM = 1
NUM_TEST_IMAGES = 1000
NUM_EXT = 0
emotionnet_dict={0: 'Sadly disgusted', 
                 1: 'Neutral',
                 2: 'Happy',
                 3: 'Sad',
                 4: 'Sad',
                 5: 'Angry',
                 6: 'Surprised',
                 7: 'Disgusted',
                 8: 'Fearful',
                 9: 'Happily surprised',
                 10: 'Happily disgusted',
                 11: 'Sadly angry',
                 12: 'Angrily disgusted',
                 13: 'Appalled',
                 14: 'Hatred',
                 15: 'Angrily surprised',
                 16: 'Sadly surprised',
                 17: 'Sadly surprised',
                 18: 'Disgustedly surprised',
                 19: 'Disgustedly surprised',
                 20: 'Disgustedly surprised',
                 21: 'Fearfully surprised',
                 22: 'Awed',
                 23: 'Sadly fearful',
                 24: 'Fearfully disgusted',
                 25: 'Fearfully angry',
}


celeba_dict = {0:'neutral', 1:'smile'}

rafd_dict={0: 'angry', 
                 1: 'disgusted',
                 2: 'fearful',
                 3: 'happy',
                 4: 'sad',
                 5: 'surprised',
                 6: 'neutral',
                 7: 'contemptuous'
}

def create_solver(opt):
    instance = Solver()
    instance.initialize(opt)
    return instance

def add_one_layer(img):
        layer = np.zeros(list(img.shape[:2]) + [1], dtype= img.dtype)
        return np.concatenate([img, layer], axis=2)

def resize_flow(flow, shape_t):
        img = add_one_layer(flow)
        output = cv2.resize(img, shape_t, interpolation=FLOW_INTER_TYPE)
        output = adjust_flow(output, img.shape[:2], shape_t)
        return output

def chw2hwc(img):
    return img.transpose((1,2,0))

def resize_img(img, shape_t):
    return cv2.resize(img, shape_t, interpolation=INTER_TYPE)

def adjust_flow(output, shape_o, shape_t):
        output[:,:,1] *= shape_t[0]*1. / shape_o[0]
        output[:,:,0] *= shape_t[1]*1. / shape_o[1]
        return output

def warp(img,  flow):
    height, width, __ = img.shape
    y_mesh, x_mesh = np.mgrid[:height,:width]
    y_coords = flow[:,:,1] + y_mesh
    x_coords = flow[:,:,0] + x_mesh
    coords = np.array([y_coords, x_coords])
    #'nearest' 0: 'biquadratic'2:bilinear 1: 'bicubic'3: 'biquartic'4: 'biquintic'5:
    im1 = sktransform.warp(img[:,:,0], coords, preserve_range=True,
                                    mode='reflect', cval=0, order=1)
    im2 = sktransform.warp(img[:,:,1], coords, preserve_range=True,
                                    mode='reflect', cval=0, order=1)
    im3 = sktransform.warp(img[:,:,2], coords, preserve_range=True,
                                    mode='reflect', cval=0, order=1)
    return np.stack([im1,im2,im3],axis=2) 



def warp_gpu(img,  flow):
    h, w, __ = img.shape
    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    xv = xv.float() / (w - 1) * 2.0 - 1
    yv = yv.float() / (h - 1) * 2.0 - 1
    grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
    flow = torch.nn.functional.upsample(flow.permute(0,3,1,2), [h,w], mode='bilinear')
    flow =  flow.permute(0,2,3,1)
    grid_x = grid + 2 * flow
    img = torch.from_numpy(img.transpose(2,0,1)).cuda().to(torch.float32).unsqueeze(0)
    res = torch.nn.functional.grid_sample(img, grid_x, mode='bilinear', padding_mode='border')
    return res.cpu().numpy().astype(np.float32)[0].transpose(1,2,0)

def numpy2im(image_numpy):
    image_numpy = (image_numpy / 2. + 0.5) * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    return Image.fromarray(image_numpy)

def make_batch(img, idx, aus_nc):
    batch_img = torch.from_numpy(np.asarray([img]))
    if FIX_TAR:
      tar_label = aus_nc - 1
    else:
      tar_label = idx % aus_nc
      if 2 == aus_nc:
        tar_label = aus_nc - 1
    batch_aus = torch.from_numpy(np.asarray([tar_label]))
    return {'src_img': batch_img, 'tar_aus': batch_aus}, tar_label

def make_batch_au(src, idx, examples, opt):
    batch_src = torch.from_numpy(np.asarray([src]))
    if opt.test_example_dir:
      if opt.test_example_type == 'sequence':
        batches = []
        tar_imgs = []
        tar_imgs_paths, tar_aus = examples['imgs'], examples['aus']
        for tar_img_path, tar_au in zip(tar_imgs_paths, tar_aus):
          tar_img = cv2.imread(tar_img_path)[:,:,::-1].transpose(2, 0 ,1).astype(np.float32)
          batch_au = torch.from_numpy(np.asarray([np.asarray(tar_au)/.5]))
          batches.append({'src_img': batch_src, 'tar_aus': batch_au})
          tar_imgs.append(tar_img)
        return batches, tar_imgs, tar_aus[-1]
      else:
        tar_imgs_paths, tar_aus = examples['imgs'], examples['aus']
        selected_idx = idx % len(tar_imgs_paths)
        tar_img = cv2.imread(tar_img_path[selected_idx])[:,:,::-1].transpose(2, 0 ,1).astype(np.float32)
        batch_au = torch.from_numpy(np.asarray([np.asarray(tar_aus[selected_idx])/.5]))
        return {'src_img': batch_src, 'tar_aus': batch_au}, tar_img, tar_aus[selected_idx]
    else:
      idx = idx%len(examples) 
      selected = examples[idx]
      aus = selected['tar_aus'].cpu().float().numpy()
      tar_imgs = selected['tar_img'].cpu().float().numpy()
      selected_idx = idx % aus.shape[0]
      print('selected_idx', selected_idx)
      selected_idx = min(selected_idx, aus.shape[0]-1)
      tar_img = (tar_imgs[selected_idx]/2. + .5 ) * 255.
      tar_aus = aus[selected_idx]
      batch_aus = torch.from_numpy(np.asarray([tar_aus]))
      return {'src_img': batch_src, 'tar_aus': batch_aus}, tar_img, tar_aus

def warp_and_resize_ms(fake, flow, residual_pyr):
    img_temp = chw2hwc(fake)
    for residual in reversed(residual_pyr):
        res_warped = warp_gpu(residual, flow)
        img_temp = resize_img(img_temp, residual.shape[:2]) + res_warped
    return img_temp

def get_hr_fake(model, residual, coef_flow, final_size):
    fake = model.fake_img.cpu().float().numpy()[0]
    #print('flow', flow.max(), flow.min())
    if isinstance(residual, list):
      fake_hr = warp_and_resize_ms(fake, model.flow*coef_flow, residual)
    else:
      tsize = residual.shape[0]
      res_warped = warp_gpu(residual, model.flow*coef_flow)
      fake_hr = resize_img(chw2hwc(fake), (tsize, tsize)) + res_warped 
    return np.clip(fake_hr, -1., 1.)

def generate_video(model, residual, nframes, test_batch, coef_flow, final_size):
    faces_list = []
    if isinstance(test_batch, list):
      for batch in test_batch:
        model.feed_batch(batch)
        #s =time.time()
        model.forward(interp_coef=1.)
        #print('time ' ,time.time()-s)
        faces_list.append([get_hr_fake(model, residual, coef_flow, final_size)])
    else:
      model.feed_batch(test_batch)
      for coef in (np.array(list(range(nframes)))/(nframes - 1. + 2.)).tolist():
        model.forward(interp_coef=coef)
        faces_list.append([get_hr_fake(model, residual, coef_flow, final_size)])
    for i in range(NUM_EXT):
      faces_list.append(faces_list[-1])
    imgs_numpy_list = []
    for face_idx in range(len(faces_list)): 
      cur_numpy = np.array(numpy2im(faces_list[face_idx][0]))
      imgs_numpy_list.extend([cur_numpy for _ in range(REPEAT_NUM)])
    return imgs_numpy_list


def make_save_path(dataset_name, basename, opt, tar_label):
    save_dir = opt.results + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if opt.control_signal_type == 'class':
      if 'emotionnet' in opt.ckpt_dir.lower():
        src_label = int(basename.split('_')[0])
        if src_label == 26:
            src_label = 0
        src_emotion = emotionnet_dict[src_label]
        tar_emotion = emotionnet_dict[tar_label]
      elif 'celeba' in opt.ckpt_dir.lower():
        src_emotion = 'unknown'
        tar_emotion = celeba_dict[tar_label]
      elif 'rafd' in opt.ckpt_dir.lower():
        src_emotion = basename.split('_')[4]
        tar_emotion = rafd_dict[tar_label]
      else:
        src_emotion = 'unknown'
        tar_emotion = 'unknown'  #celeba_dict[tar_label]
    else:
        src_emotion = 'unknown'
        tar_emotion = 'unknown' 
    if src_emotion == 'unknown' or tar_emotion == 'unknown':
      if opt.test_example_dir:
          exdr = opt.test_example_dir.split('/')[-1][1:]
      else:
          exdr = 'unknown'
      return os.path.join(save_dir, "%s-%s.gif" % (basename[:-4], exdr))
    else:
      return os.path.join(save_dir, "%s_%s-%s.gif" % (basename, src_emotion, tar_emotion))

def merge(imgs):
    img_size = imgs[0].shape[0]
    n_select = len(imgs)
    pad = img_size//25
    merged = np.ones([img_size,  (pad + img_size)*n_select + pad, 3]) * 255
    offset = 0
    for img in imgs:
        im = np.array(img)
        if im.shape[0] != img_size:
            im = cv2.resize(im, (img_size,img_size))
        merged[:, offset+pad:offset+pad+img_size, :] = im
        offset += img_size+pad
    return merged.astype(np.uint8)

def tag_example(imgs, tag_img, opt):
    tsz = imgs[0].shape[0]//4
    if isinstance(tag_img, list):
      sz = tag_img[0].shape[1]
    else:
      sz = tag_img.shape[1]
    if opt.test_example_cropped:
      left, top= 0, 0
      bottom = sz
    else:
      left = sz//4
      top = sz//4+sz//8
      bottom = top + (sz - 2 * left)
    print(len(imgs), len(tag_img))
    if isinstance(tag_img, list):
      for idx, img in enumerate(imgs):
        tar_img = tag_img[min(idx//REPEAT_NUM, len(tag_img)-1)].transpose(1,2,0)
        img[-tsz:, :tsz, :] = cv2.resize(tar_img[top:bottom,left:sz-left,:], (tsz,tsz))
    else:
      tag_img = tag_img.transpose(1,2,0)
      for img in imgs:
        img[-tsz:, :tsz, :] = cv2.resize(tag_img[top:bottom,left:sz-left,:], (tsz,tsz))
    return imgs

def get_examples(opt):
    img2au, aus = None, None
    assert os.path.exists(opt.test_example_dir + '/imgs'), print(opt.test_example_dir + '/imgs not exists')
    assert os.path.exists(opt.test_example_dir + '/aus.txt'), print('aus.txt not exists')
    with open(opt.test_example_dir + '/aus.txt', 'r') as f:
        lines = f.readlines()
        img2au = {}
        for line in lines:
            tokens = line.split()
            img_name = tokens[0]
            aus = [float(tk) for tk in tokens[1:]]
            img2au[img_name] = aus
    img_paths = sorted(glob.glob(opt.test_example_dir + '/imgs/*.jpg'))
    #img_paths = [opt.test_example_dir+'/imgs/'+str(i)+".jpg" for i in range(1, len(img_paths)+1, 1)][10:21]
    if img2au is not None:
      aus = [img2au[os.path.basename(img_path)] for img_path in img_paths]
    print(img_paths, aus)
    return {'imgs': img_paths, 'aus': aus}

class Solver(object):
    """docstring for Solver"""
    def __init__(self):
        super(Solver, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.visual = Visualizer()
        self.visual.initialize(self.opt)

    def run_solver(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks()

    def train_networks(self):
        # init train setting
        self.init_train_setting()

        # for every epoch
        for epoch in range(self.opt.epoch_count, self.epoch_len + 1):
            # train network
            self.train_epoch(epoch)
            # update learning rate
            self.cur_lr = self.train_model.update_learning_rate()
            # save checkpoint if needed

            if epoch % self.opt.save_epoch_freq == 0:
                self.train_model.save_ckpt(epoch)

        # save the last epoch 
        self.train_model.save_ckpt(self.epoch_len)

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        self.train_model = create_model(self.opt)

        self.train_total_steps = 0
        self.epoch_len = self.opt.niter + self.opt.niter_decay
        self.cur_lr = self.opt.lr

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        epoch_steps = 0
        last_print_step_t = time.time()
        for idx, batch in enumerate(self.train_dataset):

            self.train_total_steps += self.opt.batch_size
            epoch_steps += self.opt.batch_size
            # train network
            #if idx == 0:
            #    sample_batch = batch
            self.train_model.feed_batch(batch)
            self.train_model.optimize_paras(train_gen=(idx % self.opt.train_gen_iter == 0))
            # print losses

            if self.train_total_steps % self.opt.print_losses_freq == 0:
                cur_losses = self.train_model.get_latest_losses()
                avg_step_t = (time.time() - last_print_step_t) / self.opt.print_losses_freq
                last_print_step_t = time.time()
                # print loss info to command line
                info_dict = {'epoch': epoch, 'epoch_len': self.epoch_len,
                            'epoch_steps': idx * self.opt.batch_size, 'epoch_steps_len': len(self.train_dataset),
                            'step_time': avg_step_t, 'cur_lr': self.cur_lr,
                            'log_path': os.path.join(self.opt.ckpt_dir, self.opt.log_file),
                            'losses': cur_losses
                            }
                self.visual.print_losses_info(info_dict)

            # plot loss map to visdom
            if self.train_total_steps % self.opt.plot_losses_freq == 0 and self.visual.display_id > 0:
                cur_losses = self.train_model.get_latest_losses()
                epoch_steps = idx * self.opt.batch_size
                self.visual.display_current_losses(epoch - 1, epoch_steps / len(self.train_dataset), cur_losses)
            
            # display image on visdom
            if self.train_total_steps % self.opt.sample_img_freq == 0 and self.visual.display_id > 0:
                cur_vis = self.train_model.get_latest_visuals()
                self.visual.display_online_results(cur_vis, epoch)

    def test_networks(self):
        self.init_test_setting()
        if self.opt.test_dir:
            self.test_hr()
        else:
            self.test_128()

    def init_test_setting(self):
        if not self.opt.test_dir or self.opt.control_signal_type == 'au':
            self.test_dataset = create_dataloader(self.opt)
        self.test_model = create_model(self.opt)

    def img_normalizer(self):
        transform_list = []
        transform_list.append(transforms.Resize([self.opt.test_size, self.opt.test_size], Image.BICUBIC))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        img2tensor = transforms.Compose(transform_list)
        return img2tensor

    def get_img_by_path(self, img_path):
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def get_temp_vis(self, residual, src_hr, tar_img):
        if isinstance(tar_img, list):
            tar_img = tar_img[-1]
        tsize = self.opt.test_size
        self.test_model.forward(interp_coef=1.)
        fake = self.test_model.fake_img.cpu().float().numpy()[0]
        flow = self.test_model.flow.cpu().float().numpy()[0]
        fake_warped = self.test_model.img_warp.cpu().float().numpy()[0]
        flow_hr = resize_flow(flow, (tsize, tsize))[:,:,:2]
        res_warped = warp(residual, flow_hr) if residual is not None else None
        fake_hr_blur = resize_img(chw2hwc(fake), (tsize, tsize))
        fake_hr = res_warped + fake_hr_blur if res_warped is not None else fake_hr_blur
        fake_hr = np.clip(fake_hr, -1., 1.)
        selected = [np.array(self.visual.numpy2im(src_hr.numpy()))]
        if self.opt.control_signal_type == 'au':
          selected.append(Image.fromarray(tar_img.astype(np.uint8).transpose(1,2,0)))
        selected.append(np.array(self.visual.numpy2im(cvb.flow2rgb(flow_hr), no_transpose=True)))
        selected.append(np.array(self.visual.numpy2im(fake_warped)))
        selected.append(np.array(self.visual.numpy2im(fake_hr_blur, no_transpose=True)))
        if res_warped is not None:
          selected.append(np.array(self.visual.numpy2im(res_warped, no_transpose=True)))
        selected.append(np.array(self.visual.numpy2im(fake_hr, no_transpose=True)))
        return selected


    def resize_ave(self, src_hr):
        to_size = self.opt.final_size
        img_src = src_hr.numpy()
        c, h, w = img_src.shape
        img = img_src.reshape(c, to_size, h//to_size, to_size, w//to_size)
        img = img.mean(axis=(2,4))
        img = img.reshape((c, to_size, to_size))
        upsampled = cv2.resize(chw2hwc(img), (w,h), interpolation=INTER_TYPE)
        residual = chw2hwc(img_src) - upsampled
        return img, residual, upsampled


    def resize_ave_ms(self, src_hr):
        to_size = self.opt.final_size
        img_src = src_hr.numpy()
        c, h, w = img_src.shape

        temp_size = h
        img = img_src
        imgs_pyr = [img]
        while temp_size > self.opt.final_size:
          temp_size = temp_size//2
          img = img.reshape(c, temp_size, 2, temp_size, 2)
          img = img.mean(axis=(2,4))
          img = img.reshape((c, temp_size, temp_size))
          imgs_pyr.append(img)
        imgs_pyr = [chw2hwc(im) for im in imgs_pyr]
        upsampled_pyr = []
        residual_pyr = []
        for i, im in enumerate(imgs_pyr):
          if i == len(imgs_pyr) -1 :
              break
          im2 = imgs_pyr[i+1]
          upsampled = cv2.resize(im2, im.shape[:2], interpolation=INTER_TYPE)
          upsampled_pyr.append(upsampled)
          residual_pyr.append(im - upsampled)
        return img, residual_pyr, upsampled_pyr

    def get_examples_from_data_root(self, num):
        examples = []
        for batch_idx, batch in enumerate(self.test_dataset):
            if batch_idx >= num:
                break
            examples.append(batch)
        return examples
            
    def test_hr(self):
        import imageio
        dataset_name = '_'.join(self.opt.test_dir.split('/')[-3:]) + '/'
        if self.opt.use_multiscale:
            dataset_name += '_ms'
        if self.opt.save_test_gif:
            dataset_name += '_gif'
        img_paths = glob.glob(self.opt.test_dir + '/*[g|G]')
        if self.opt.control_signal_type == 'au':
          if self.opt.test_example_dir:
            if self.opt.test_example_type == 'sequence':
              exemples = get_examples(self.opt)
            else:
              exemples = get_examples(self.opt)
          else:
            exemples = self.get_examples_from_data_root(20)
        with torch.no_grad():
            for idx, p_src in enumerate(img_paths[0:NUM_TEST_IMAGES]):
              print("processing %s"%(p_src))
              src_hr = self.get_img_by_path(p_src)
              self.opt.test_size = (src_hr.size[1]//self.opt.final_size) * self.opt.final_size
              self.normalize_img = self.img_normalizer()
              src_hr = self.normalize_img(src_hr)
              if self.opt.use_multiscale:
                src_lr, residual_pyr, upsampled_pyr = self.resize_ave_ms(src_hr)
              else:
                src_lr, residual, upsampled = self.resize_ave(src_hr)
              if  self.opt.control_signal_type == 'class':
                  test_batch, tar_label = make_batch(src_lr, idx, self.opt.aus_nc)
              elif self.opt.control_signal_type == 'au':
                test_batch, tar_img, tar_label = make_batch_au(src_lr, idx, exemples, self.opt)

              basename = os.path.basename(p_src)
              s = time.time()
              if self.opt.use_multiscale:
                imgs_list = generate_video(self.test_model, residual_pyr, self.opt.nframes, test_batch, self.opt.coef_flow, self.opt.final_size)
              else:
                imgs_list = generate_video(self.test_model, residual, self.opt.nframes, test_batch, self.opt.coef_flow, self.opt.final_size)
              print('time: ' , time.time() - s)
              save_path = make_save_path(dataset_name, basename, self.opt, tar_label)
              if self.opt.save_test_gif:
                if self.opt.control_signal_type == 'au':
                    tag_example(imgs_list, tar_img, self.opt)
                imageio.mimsave(save_path, imgs_list)
              else:
                if self.opt.save_temp_results:
                  if self.opt.use_multiscale:
                      residual = residual_pyr[0] if len(residual_pyr) > 0 else None
                  if not self.opt.control_signal_type == 'au':
                      tar_img = None
                  selected = self.get_temp_vis(residual, src_hr, tar_img)
                  res = merge(selected)
                  Image.fromarray(res).save(save_path+'_temp.jpg')
                end = len(imgs_list)
                selected = [imgs_list[int(idx)] for idx in np.arange(0, end, end/5)]
                if self.opt.control_signal_type == 'au':
                    if isinstance(tar_img, list):
                        tar_img = tar_img[-1]
                    selected.append(Image.fromarray(tar_img.astype(np.uint8).transpose(1,2,0)))
                res = merge(selected)
                Image.fromarray(res).save(save_path+'seq.jpg')
                
    def test_128(self):
        for batch_idx, batch in enumerate(self.test_dataset):
            with torch.no_grad():
                # interpolate several times
                faces_list = [batch['src_img'].float().numpy()]
                paths_list = [batch['src_path'], batch['tar_path']]
                test_batch = {'src_img': batch['src_img'], 'tar_aus': batch['tar_aus'], 'src_aus':batch['src_aus'], 'tar_img':batch['tar_img']}

                self.test_model.feed_batch(test_batch)
                nframes = 10
                for coef in (np.array(list(range(nframes)))/(nframes - 1.)).tolist():
                  self.test_model.forward(interp_coef=coef)
                  #faces_list.append(self.test_model.img_warp.cpu().float().numpy())
                  faces_list.append(self.test_model.fake_img.cpu().float().numpy())
                faces_list.append(batch['tar_img'].float().numpy())
            self.test_save_imgs(faces_list, paths_list)

    def test_save_imgs(self, faces_list, paths_list):
        for idx in range(len(paths_list[0])):
            src_name = os.path.splitext(os.path.basename(paths_list[0][idx]))[0]
            tar_name = os.path.splitext(os.path.basename(paths_list[1][idx]))[0]

            if self.opt.save_test_gif:
                import imageio
                imgs_numpy_list = []
                for face_idx in range(len(faces_list) - 1):  # remove target image
                    cur_numpy = np.array(self.visual.numpy2im(faces_list[face_idx][idx]))
                    imgs_numpy_list.extend([cur_numpy for _ in range(3)])
                saved_path = os.path.join(self.opt.results, "%s_%s.gif" % (src_name, tar_name))
                imageio.mimsave(saved_path, imgs_numpy_list)
            else:
                # concate src, inters, tar faces
                concate_img = np.array(self.visual.numpy2im(faces_list[0][idx]))
                for face_idx in range(1, len(faces_list)):
                    concate_img = np.concatenate((concate_img, np.array(self.visual.numpy2im(faces_list[face_idx][idx]))), axis=1)
                concate_img = Image.fromarray(concate_img)
                # save image
                saved_path = os.path.join(self.opt.results, "%s_%s.jpg" % (src_name, tar_name))
                concate_img.save(saved_path)

            print("[Success] Saved images to %s" % saved_path)


