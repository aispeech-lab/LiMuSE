from turtle import done
from numpy.core.fromnumeric import clip
import torch
import time
import os
import sys
import numpy as np
from torch import cuda
from torch.autograd import Variable
from utils import cal_using_wav
from LiMuSE import check_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from loss_func import cal_sisnr_order_loss
from anybit import QuaOp
import torch.nn as nn
from tensorboardX import SummaryWriter


def to_device(dicts, device):

    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        elif isinstance(datas, list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')

class Trainer():

    def __init__(self,
                 net,
                 checkpoint="checkpoint",
                 model_save_path='model_save_path',
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 stop=6,
                 num_epochs=100,
                 QA_flag=False,
                 ak=8, # bits for activation
                 bit=3, # bits for weights
                 temperature=10,
                 log_path='log', # save path for the bias of quantization functions
                 tensorboard='path'): 
        self.writer = SummaryWriter(log_dir=tensorboard, flush_secs=60)
        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        if model_save_path and not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.model_save_path = model_save_path

        # build the logger object
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # current epoch
        self.stop = stop
        self.updates = 0
        self.updates_eval = 0
        self.temperature = temperature

        # Whether to resume the model
        if resume['resume_state']:
            if not os.path.exists(resume['path']):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = torch.load(resume['path'], map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            print("Resume from checkpoint {}: epoch {:d}".format(
                resume['path'], self.cur_epoch))
            # load nnet
            net.load_state_dict(cpt["model_state_dict"])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
            # self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
            
        self.optimizer_alpha = self.create_optimizer(optimizer, optimizer_kwargs)
        self.optimizer_beta = self.create_optimizer(optimizer, optimizer_kwargs)

        # check model parameters
        self.param = check_parameters(self.net)

        # Reduce lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience, verbose=True, min_lr=min_lr)

        # logging
        print("Starting preparing model ............")
        print("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.param))
        self.clip_norm = clip_norm
        # clip norm
        if clip_norm:
            print(
                "Gradient clipping by {}, default L2".format(clip_norm))

        # number of epoch
        self.num_epochs = num_epochs

        # quantization
        self.QA_flag = QA_flag
        self.ak = ak
        self.bit = bit
        self.alpha = []
        self.beta = []
        self.init_T = 0
        self.curr_T = 0

        if QA_flag:
            # count the quantized modules
            def count_modules(module_list):
                count_targets = 0
                for submodel in module_list:
                    for m in submodel.modules():
                        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear): # 20220726, @vyouman，去掉ConvTransposed1d
                            count_targets = count_targets + 1
                return count_targets

            # Need to manually pass in quantized_module_lists
            quantized_module_lists = [self.net.encoder, self.net.voiceprint_encoder, self.net.visual_encoder, self.net.context_enc_1, self.net.context_enc_2,
                                      self.net.audio_block, self.net.fusion_block, self.net.context_dec_1, self.net.context_dec_2, self.net.gen_masks]

            count = count_modules(quantized_module_lists)
            for i in range(count):
                self.alpha.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))
                self.beta.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))


            if self.bit == 1:
                qw_values = [-1, 1]
            else:
                qw_values = list(range(-2**(self.bit-1)+1, 2**(self.bit-1)))
                n = len(qw_values) - 1
                
            numpy_file = os.path.join(log_path, 'bias.npy')
            if not os.path.exists(numpy_file):
                print('Creating numpy file', numpy_file)
                QW_biases = []
                initialize_biases = True
            else:
                print('Loading biases from numpy file', numpy_file)
                initialize_biases = False
                QW_biases = np.load(numpy_file)
                QW_biases = list(QW_biases)
                print('QW_bias of numpy file', QW_biases)


            assert quantized_module_lists is not None
            self.qua_op = QuaOp(quantized_module_lists, QW_biases=QW_biases, QW_values=qw_values, initialize_biases=initialize_biases, init_linear_bias=0) 


            if resume['resume_state_Q']:
                cpt = torch.load(resume['path'], map_location="cpu")
                self.init_T = cpt['T']
                self.curr_T = cpt['T']
                self.temperature = cpt['temperature']
                self.updates = cpt['updates']
                self.updates_eval = cpt['updates_eval']
                print("Resume temperature: {}".format(self.init_T))
                self.optimizer_alpha = self.create_optimizer(
                    optimizer, optimizer_kwargs, state=cpt["optim_state_dict_alpha"])
                self.optimizer_beta = self.create_optimizer(
                    optimizer, optimizer_kwargs, state=cpt["optim_state_dict_beta"])
                cptelse = torch.load(resume['path'],map_location=lambda storage, loc: storage.cuda(0))
                self.alpha = cptelse["alpha"]
                self.beta = cptelse['beta']

            if initialize_biases:
                # print('Save and freeze bias for quantization!')
                numpy_file = os.path.join(log_path, 'bias.npy')
                np.save(numpy_file, self.qua_op.QW_biases)
                print('Saving the bias in', numpy_file)
        
            # quantized modules
            param_count = 0
            for name, param in self.net.named_parameters():
                if param.requires_grad:
                    # print('Name, param', name, param.numel())
                    param_count += param.numel()
            print('parameter number: %d\n' % param_count)

            ssmodel_num = 0
            for submodel in quantized_module_lists:
                # print('submodel', submodel)
                for name, param in submodel.named_parameters():
                    if param.requires_grad:
                        # print('Name, param', name, param.numel())
                        ssmodel_num += param.numel()
            print('ssmodel module parameters num:%d' % ssmodel_num)
            print('Quantized parameters count:%d' % self.qua_op.param_num)
            print('Quantized module count:%d' % count)
            not_quantized_num = param_count - self.qua_op.param_num
            ideal_quantized_size = (not_quantized_num + count * 2) * 32 / 8 / 1024 / 1024 + self.qua_op.param_num * self.bit / 8 / 1024 / 1024 
            ideal_full_precision_model_size = param_count * 32 / 8 / 1024 / 1024
            print('ideal model size for full-precision: %.3f\n' % ideal_full_precision_model_size)
            print('Ideal Quantized model size:%.3f' %ideal_quantized_size)
            print('Ideal compression ratio %.3f' % (ideal_full_precision_model_size / ideal_quantized_size))



    def create_optimizer(self, optimizer, kwargs, state=None):

        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        print("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            print("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, best=True):

        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "T": self.curr_T,
                "temperature": self.temperature,
                "optim_state_dict_alpha": self.optimizer_alpha.state_dict(),
                "optim_state_dict_beta": self.optimizer_beta.state_dict(),
                "alpha": self.alpha,
                "beta": self.beta,
                "updates": self.updates, 
                'updates_eval': self.updates_eval
            },
            os.path.join(self.checkpoint,
                         "{0}.pt".format("best" if best else "last")))

    def reg_save_checkpoint(self):

        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "T": self.curr_T,
                "temperature": self.temperature,
                "optim_state_dict_alpha": self.optimizer_alpha.state_dict(),
                "optim_state_dict_beta": self.optimizer_beta.state_dict(),
                "alpha": self.alpha,
                "beta": self.beta,
                "updates": self.updates, 
                'updates_eval': self.updates_eval
            },
            os.path.join(self.checkpoint,
                         "{0}.pt".format(self.cur_epoch)))
    
    def save_model(self, best=True):
        print('Saved a better model ......')
        torch.save(self.net, 
                    os.path.join(self.model_save_path,
                        "{0}.pt".format("best_model" if best else "last_model")))

    def train(self, grid_samples):
        print('Training model ......')
        self.net.train()
        step = 0
        losses = []
        start = time.time()
        train_dataloader = grid_samples.get_samples(phase='train')
        # Increase T per epoch
        if self.QA_flag:
            if self.init_T == 0:
                self.curr_T = self.init_T + self.temperature
            else:
                self.curr_T += self.temperature
            print('The current temperature is', self.curr_T)
            
        while True:
            train_data = train_dataloader.__next__()
            if train_data == False:
                break
            train_data = to_device(train_data, self.device)
            self.optimizer.zero_grad()

            # quantization
            if self.QA_flag:
                self.optimizer_alpha.zero_grad()
                self.optimizer_beta.zero_grad()
                if self.init_T == 0: 
                    init = True
                else:
                    init = False
                self.qua_op.quantization(self.curr_T, self.alpha, self.beta, init=init) # init_T为0的时候初始化alpha和beta

            predict_wav = data_parallel(self.net,(train_data['mix_wav'], train_data['gt_ref_vp'], train_data['gt_visual']), device_ids=self.gpuid)
            True_wav = train_data['spk1_wav']
            shape = True_wav.shape
            True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
            True_wav = True_wav.unsqueeze(1)
            loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
            loss.backward()

            # quantization: restore
            if self.QA_flag:
                self.qua_op.restore_params()
                alpha_grad, beta_grad = self.qua_op.updateQuaGradWeight(self.curr_T, self.alpha, self.beta, init=init)
                for idx in range(len(self.alpha)):
                    self.alpha[idx].grad = Variable(torch.FloatTensor([alpha_grad[idx]]).cuda())
                    self.beta[idx].grad = Variable(torch.FloatTensor([beta_grad[idx]]).cuda())

                self.optimizer_alpha.step()
                self.optimizer_beta.step()

            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses += [loss.item()]

            # logging
            self.writer.add_scalar('Training loss/iter', loss.item(), self.updates)
            self.updates += 1

            step += 1
            if step % 200 == 0:
                avg_loss = sum(losses[-self.logging_period:]) / self.logging_period
                print('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))

            if self.QA_flag:
                # update init_T
                self.init_T = self.curr_T

        total_loss_avg = np.array(losses).mean()
        end = time.time()
        print('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
                self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end - start) / 60))
        return total_loss_avg

    def val(self, grid_samples):
        print('Validation model ......')
        self.net.eval()
        losses = []
        SDR_SUM = np.array([])
        SDRi_SUM = np.array([])
        step = 0
        start = time.time()
        val_dataloader = grid_samples.get_samples(phase='valid')
        with torch.no_grad():
            while True:
                val_data = val_dataloader.__next__()
                if val_data == False:
                    break
                val_data = to_device(val_data, self.device)

                # quantization
                if self.QA_flag:
                    self.qua_op.quantization(self.curr_T, self.alpha, self.beta, init=False, train_phase=False)

                predict_wav = data_parallel(self.net, (val_data['mix_wav'], val_data['gt_ref_vp'], val_data['gt_visual']), device_ids=self.gpuid)
                True_wav = val_data['spk1_wav']
                shape = True_wav.shape
                True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
                True_wav = True_wav.unsqueeze(1)
                loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
                losses += [loss.item()]
                step += 1

                # logging
                self.writer.add_scalar('CV loss/iter', loss.item(), self.updates_eval)
                self.updates_eval += 1

                try:
                    mix = val_data['mix_wav']
                    mix = mix[:,0,:]
                    predict = torch.squeeze(predict_wav, dim=1)
        
                    sdr_aver_batch, sdri_aver_batch = cal_using_wav(
                        2, mix, val_data['spk1_wav'], predict)
                    self.writer.add_scalar('CV loss/SDR', sdr_aver_batch, self.updates_eval)
                    SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
                    SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
                except AssertionError as wrong_info:
                    print('Errors in calculating the SDR: %s' % wrong_info)

                # quantization restore
                if self.QA_flag:
                    self.qua_op.restore_params()
                if step % 200 == 0:
                    avg_loss = sum(losses[-self.logging_period:]) / self.logging_period
                    print('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.cur_epoch, step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
            total_loss_avg = np.array(losses).mean()
        end = time.time()        
        print('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end - start) / 60))
        print('SDR_aver_now: %f' % SDR_SUM.mean())
        print('SDRi_aver_now: %f' % SDRi_SUM.mean())               
        return total_loss_avg

    def test(self, grid_samples):
        print('Testing model ......')
        self.net.eval()
        losses = []
        SDR_SUM = np.array([])
        SDRi_SUM = np.array([])
        step = 0
        start = time.time()
        val_dataloader = grid_samples.get_samples(phase='test')
        with torch.no_grad():
            while True:
                val_data = val_dataloader.__next__()
                if val_data == False:
                    break
                val_data = to_device(val_data, self.device)

                # quantization
                if self.QA_flag:
                    self.qua_op.quantization(self.curr_T, self.alpha, self.beta, init=False, train_phase=False)

                predict_wav = data_parallel(self.net, (val_data['mix_wav'], val_data['gt_ref_vp'], val_data['gt_visual']), device_ids=self.gpuid)
                True_wav = val_data['spk1_wav']
                shape = True_wav.shape
                True_wav_len = Variable(torch.from_numpy(np.zeros((shape[0], 1), 'int32') + shape[1])).cuda()
                True_wav = True_wav.unsqueeze(1)
                loss = cal_sisnr_order_loss(True_wav, predict_wav, True_wav_len)
                losses += [loss.item()]
                step += 1

                # logging
                self.writer.add_scalar('CV loss/iter', loss.item(), self.updates_eval)
                self.updates_eval += 1

                try:
                    mix = val_data['mix_wav']
                    mix = mix[:,0,:]
                    predict = torch.squeeze(predict_wav, dim=1)
        
                    sdr_aver_batch, sdri_aver_batch = cal_using_wav(
                        1, mix, val_data['spk1_wav'], predict)
                    self.writer.add_scalar('CV loss/SDR', sdr_aver_batch, self.updates_eval)
                    SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
                    SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
                except AssertionError as wrong_info:
                    print('Errors in calculating the SDR: %s' % wrong_info)

                # quantization restore
                if self.QA_flag:
                    self.qua_op.restore_params()
                if step % 200 == 0:
                    avg_loss = sum(losses[-step:]) / step
                    print('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.cur_epoch, step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
            total_loss_avg = np.array(losses).mean()
        end = time.time()        
        print('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end - start) / 60))
        print('SDR_aver_now: %f' % SDR_SUM.mean())
        print('SDRi_aver_now: %f' % SDRi_SUM.mean())               
        return total_loss_avg

    def run(self, grid_samples):
        train_losses = []
        val_losses = []
        with torch.cuda.device(self.gpuid[0]):
            stats = dict()
            self.reg_save_checkpoint()
            self.save_checkpoint(best=False)
            self.save_model(best=False)
            val_loss = self.val(grid_samples)
            # output = self.inference(grid_samples)
            best_loss = val_loss
            print("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0

            self.scheduler.best = best_loss
            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                train_loss = self.train(grid_samples)
                val_loss = self.val(grid_samples)
                # Tensorboard
                t_loss = np.array(train_loss)
                v_loss = np.array(val_loss)
                self.writer.add_scalar('Training_loss/Epoch', t_loss, self.cur_epoch)
                self.writer.add_scalar('Val_loss/Epoch', v_loss, self.cur_epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                self.reg_save_checkpoint()
                if val_loss > best_loss:
                    no_impr += 1
                    print('no improvement, best loss: {:.4f}'.format(self.scheduler.best))

                    # 20220804
                    if no_impr == self.patience:
                        # reset!            
                        reset_path = os.path.join(self.checkpoint,"{0}.pt".format("best"))
                        cpt = torch.load(reset_path, map_location="cuda:0")
                        self.cur_epoch = cpt["epoch"]
                        print("Reset from checkpoint {}: epoch {:d}".format(reset_path, self.cur_epoch))
                        # load nnet
                        self.net.load_state_dict(cpt["model_state_dict"])

                        self.curr_T = cpt['T']
                        self.alpha = cpt["alpha"]
                        self.beta = cpt['beta']
                        self.temperature = cpt['temperature']
                        self.updates = cpt['updates']
                        self.updates_eval = cpt['updates_eval']
                        print("Reset temperature: {}".format(self.curr_T))  
                        self.net.to(self.device) 
                else:
                    best_loss = val_loss
                    no_impr = 0
                    self.save_checkpoint(best=True)
                    print('Epoch: {:d}, now best loss change: {:.4f}'.format(self.cur_epoch, best_loss))
                    self.save_model(best=True)
                # schedule here
                self.scheduler.step(val_loss)
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.stop:
                    print(
                        "Stop training cause no impr for {:d} epochs".format(no_impr))
                    break
            test_loss = self.test(grid_samples)
            print("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.num_epochs))
