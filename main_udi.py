# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
UDI training.

Following the public repo of DINO:
https://github.com/facebookresearch/dino/blob/main/main_dino.py
"""


import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from functools import partial

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead, MaskedAttention
from udi_augmentation import DataAugmentation

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():

    parser = argparse.ArgumentParser('UDI', add_help=False)
    
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_small', 'vit_base', 'vit_large'] \
                + torch.hub.list("facebookresearch/deit:main"), 
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_small with patch size of 16.""")
    
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory by cubic complexity.""") 
    
    parser.add_argument('--out_dim', default=65536, type=int, 
        choices=[2048, 4096, 8192, 16384, 32768, 65536, 131072],
        help="""Dimensionality of the head output. Larger values work better for complex 
        and large datasets.""")
    
    ######
    parser.add_argument('--include_patch', default=True, type=utils.bool_flag,
        help="""Include patch representation distillation in the training pipeline.""")
    
    parser.add_argument('--number_cls_token', default=2, type=int,
        help="""Number of class token in ViT.""")
    #######
    
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of prediction head.
        Not normalizing leads to better performance but can make the training unstable.
        It is typically to set to False with vit_small and True with larger models.""")
    
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="""Whether to use batch normalizations in projection head.""")
    
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with 
        cosine schedule. We recommend setting a higher value with small batches: for example 
        use 0.9995 with batch size of 256.""")

    parser.add_argument('--alpha', default=0.5, type=float, help="""Blending factor for the 
        predictions  on image-level and patch level. We recommend setting a higher value if 
        training is unstable.""")
    
    
    
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
            help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
    
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""End value
        of the teacher temperature (after linear warmup). Following DINO, value above 0.07 is 
        not recommended as it leads to unstable training. We recommend starting with the default 
        value of 0.04 and increase this slightly if needed.""")
    
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    
    
    
    # Training/Optimization parameters, mainly following the setting in DINO
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether 
        or not to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, reducing the patch size or using bigger ViTs.
        If --patch_size is set less than 16 (e.g. 8), we recommend disabling mixed precision training 
        (--use_fp16 false) to avoid unstabilities.""")
    
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    
    parser.add_argument('--weight_decay_end', type=float, default=0.1, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU, e.g., 1024/16=64')
    
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs of training.')
    
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during the first epoch 
        helps training. Try increasing this value if the loss does not decrease.""")
    
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""") # 0.0008
    
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    
    parser.add_argument('--min_lr', type=float, default=1e-5, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. AdamW is recommended for ViTs.""")
    
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.25, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping for teacher model. When disabling multi-crop, 
        a wider range of scale ("--global_crops_scale 0.15 1." for example) is recommended.""")
    
    parser.add_argument('--local_crops_number', type=int, default=6, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    parser.add_argument('--global_sampling_window_size', type=int, default=3,
        help="""Window size of stratified random sampling for global crops.""")

    parser.add_argument('--local_sampling_window_size', type=int, default=2,
        help="""Window size of stratified random sampling for global crops.""")

    
    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
            help='Please specify path to the ImageNet training data.')
    
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    
    return parser


def train_udi(args):
    
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        sampling_window_size_g = args.global_sampling_window_size,
        sampling_window_size_l = args.local_sampling_window_size,
        patch_size = args.patch_size,
        horizontal_flip=0.5,
    )
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    num_class_token = args.number_cls_token
    if not args.include_patch:
        num_class_token=1
        
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            num_class_token=num_class_token,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](
            num_class_token=num_class_token,
            patch_size=args.patch_size)
        embed_dim = student.embed_dim
        
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unsupported architecture: {args.arch}")


    # Prediction heads:
    student_head = DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,)
    teacher_head = DINOHead(
        embed_dim,
        args.out_dim, 
        args.use_bn_in_head,)

    if args.include_patch:
        # udi wrapper to append SRS module for patch-level representations
        student = utils.UDIWrapper(student, 
                                attn=MaskedAttention(dim=embed_dim, num_heads=6, qkv_bias=True))
        teacher = utils.UDIWrapper(teacher,
                                attn=MaskedAttention(dim=embed_dim, num_heads=6, qkv_bias=True))
    else:
        student = utils.UDIWrapper(student, include_cls=True)
        teacher = utils.UDIWrapper(teacher, include_cls=True)
    
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        student_head,
        num_local_crops=args.local_crops_number,
        include_patch=args.include_patch,
        local_patch=False,
        decouple_head=False)

    teacher = utils.MultiCropWrapper(
        teacher,
        teacher_head, 
        num_local_crops=args.local_crops_number,
        include_patch=args.include_patch,
        local_patch=False,
        decouple_head=False)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
        
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    # teacher temperature scheduler
    teacher_temp_schedule = np.concatenate(
        (np.linspace(args.warmup_teacher_temp, args.teacher_temp, args.warmup_teacher_temp_epochs), 
         np.ones(args.epochs - args.warmup_teacher_temp_epochs) * args.teacher_temp))

    udi_loss = UDILoss(
        args.local_crops_number,
        args.out_dim,
        teacher_temp_schedule=teacher_temp_schedule,
        alpha=args.alpha,
        num_clt = args.number_cls_token,
        student_temp=0.1,
        center_momentum=0.9,
        include_patch=args.include_patch
    ).cuda()
    

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
        
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        udi_loss=udi_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting UDI training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of UDI ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, udi_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'udi_loss': udi_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, udi_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (data, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images and maps to gpu
        images = [im.cuda(non_blocking=True) for im in data['crops']]
        src_maps = [map.cuda(non_blocking=True) for map in data['src_maps']] if args.include_patch else None
        tgt_maps = [map.cuda(non_blocking=True) for map in data['tgt_maps']] if args.include_patch else None
        
        # teacher and student forward passes + compute udi loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            
             # only the 2 global views pass through the teacher
            teacher_output = teacher.forward(images[:2], tgt_maps, branch=1)
            student_output = student.forward(images, src_maps)
            loss, loss_spec = udi_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, {}, stopping training".format(loss.item(), str(loss_spec)), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        if args.include_patch and args.number_cls_token==2:
            metric_logger.update(g_cl=loss_spec[0][0].item())
            metric_logger.update(g_mcl=loss_spec[0][1].item())
            metric_logger.update(g_pch=loss_spec[0][2].item())
            metric_logger.update(l_cl=loss_spec[1][0].item())
            metric_logger.update(l_mcl=loss_spec[1][1].item())
        elif args.include_patch and args.number_cls_token==1:
            metric_logger.update(g_cl=loss_spec[0][0].item())
            metric_logger.update(g_pch=loss_spec[0][1].item())
            metric_logger.update(l_cl=loss_spec[1][0].item())
        else:
            metric_logger.update(g_cl=loss_spec[0][0].item())
            metric_logger.update(l_cl=loss_spec[1][0].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class Centering(nn.Module):
    def __init__(self,
                 num_centers,
                 out_dim=65536, 
                 center_momentum = 0.9
                ):
        super().__init__()
        self.num_centers = num_centers
        self.center_momentum=center_momentum
        # centers are non-trainable parameters
        self.register_buffer("center", torch.zeros(num_centers, out_dim))
        self.register_buffer("count", torch.zeros(num_centers,1))
            
    def __getitem__(self, i):
        return partial(self.forward, center=self.center[i])
        
    def __len__(self):
        return self.center.shape[0]
        
    def forward(self, x, temp, center):
        #center: 1, 1, dim
        x = F.softmax((x - center) / temp, dim=-1)
        return x

    @torch.no_grad()    
    def update_center(self, teacher_output, masks):
        """
        Update center used for teacher output.
        """
        batch_cls_center = torch.zeros_like(self.center)
        count =  torch.zeros_like(self.count)
        for pred, mask, n in zip(teacher_output, masks, [0,1]):
            for i, (p, m) in enumerate(zip(pred, mask)):
                if m is None:
                    if n==1 and i==0:
                        continue
                    count[i] += p.flatten(0,-2).shape[0]
                    batch_cls_center[i] += p.flatten(0,-2).sum(dim=0)
                else:
                    count[i] += m.sum()
                    batch_cls_center[i] += (p*m.squeeze(0)).flatten(0,-2).sum(dim=0)
        batch_cls_center /= count
        dist.all_reduce(batch_cls_center)
        batch_cls_center /= dist.get_world_size()#count
        #or
        #dist.all_reduce(batch_cls_center)
        #dist.all_reduce(count)
        #batch_cls_center /= count
        self.center = self.center * self.center_momentum + batch_cls_center * (1 - self.center_momentum)


class UDILoss(nn.Module):
    def __init__(self,  
                 num_local_crops,
                 dim,
                 teacher_temp_schedule,
                 include_cls=True,
                 include_patch=True,
                 num_clt=2,
                 alpha=0.5,
                 student_temp=0.1,
                 center_momentum=0.9,
                 weights=None):
        super().__init__()
        self.student_temp = student_temp
        self.num_clt = num_clt
        self.dim = dim
        self.num_local_crops = num_local_crops
        self.teacher_temp_schedule = teacher_temp_schedule
        self.alpha = alpha
        self.include_cls = include_cls
        self.include_patch = include_patch
        self.num_ctr = num_clt + include_patch
        self.weights=weights or [1/self.num_ctr] * self.num_ctr
        self.centering = Centering(self.num_ctr, 
                                   center_momentum=center_momentum, 
                                   out_dim=dim)

    def cross_entropy(self, x, y, mask=None):
        if mask is not None:
            ce = torch.sum(-y * F.log_softmax(x.masked_fill(mask.squeeze(0).bool().logical_not(), 1e-9), dim=-1), dim=-1)
            #ce = torch.sum(-y * F.log_softmax(x + 1e-9, dim=-1), dim=-1)
            ce = torch.sum(ce * mask.squeeze()) / mask.sum()
        else:
            ce = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
            ce = ce.mean()
        return ce
        
    def data_preparation(self, teacher_output):
        """1. Align the shape 
           2. Construct tgt distribution for the second cls token"""
        
        p, m = teacher_output
        nl = self.num_local_crops
        
        if self.include_cls:
            p[1][0] = p[0][0]
            p[1][0] = p[1][0].reshape(2, -1, self.dim).transpose(0,1).repeat(nl,1,1)
            
            #construct multi-modal target
            if self.num_clt == 2 and self.include_patch:
                p[0][1] = torch.sum(p[0][2] * m[0][2] / (m[0][2].sum(-2, True) + 1e-5), -2, True)
                p[1][1] = torch.sum(p[1][2] * m[1][2] / (m[1][2].sum(-2, True) + 1e-5), -2, True)
                p[1][1] = p[1][1].reshape(nl, 2, -1, self.dim).transpose(1,2).reshape(-1, 2, self.dim)
                p[0][1] = self.alpha * p[0][1].squeeze(0) + (1 - self.alpha) * p[0][0]
                p[1][1] = self.alpha * p[1][1] + (1 - self.alpha) * p[1][0]
                
        if self.include_patch:
            p[1][-1] = p[1][-1].reshape(nl, 2, -1, *p[1][-1].shape[-2:]).transpose(0,1).flatten(1,2)
            m[1][-1] = m[1][-1].reshape(nl, 2, -1, *m[1][-1].shape[-2:]).transpose(0,1).flatten(1,2)
        return [p,m]

            
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        output: [pred, mask]
        pred: [global, local]   #mask: [global, local]
        global/local: [cls1, cls2, ph]
        """
        pred_s, mask_s = student_output
        pred_t, mask_t = self.data_preparation(teacher_output)
        out_t = [[], []]
        
        # prediction centering
        temp = self.teacher_temp_schedule[epoch]
        for i in range(2):
            j_s =len(pred_s[i])
            for j in range(self.num_ctr):
                if j < j_s:
                    pred_s[i][j] = pred_s[i][j].squeeze(0) / self.student_temp
                out_t[i].append(pred_t[i][j].detach())
                pred_t[i][j] = self.centering[j](out_t[i][j], temp).squeeze(0)

        # loss calculation:
        total_loss = 0
        
        loss = []
        for pt, ps, ms in zip(pred_t, pred_s, mask_s):
            sub_loss = []
            for t, s, m in zip(pt, ps, ms):
                sub_loss.append(self.cross_entropy(s, t, m))
            loss.append(sub_loss)
            
        # weighted sum of global and local crops:
        # weighted sum of global and local crops:
        loss_per_level=[]
        loss[1] += [-1]*(len(loss[0])-len(loss[1]))
        for l0, l1 in zip(loss[0], loss[1]):
            l = l0 * 2 + l1 * 2 * self.num_local_crops * int(l1>=0)
            l /= 2 + 2 * self.num_local_crops * int(l1>=0)
            loss_per_level.append(l )
        total_loss = sum([l*w for l, w in zip(loss_per_level, self.weights)])
        
        self.centering.update_center(out_t, mask_t)
        
        return total_loss, loss
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('UDI', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_udi(args)