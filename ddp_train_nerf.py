#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim
import torch.distributed
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo
import time
from data_loader_split import load_event_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER
import logging
import json
import time
from torch.profiler import profile, record_function, ProfilerActivity
from nerf_sample_ray_split import CameraManager

logger = logging.getLogger(__package__)


def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # create formatter
    formatter = logging.Formatter(f'%(asctime)s [{bold_red}%(levelname)s{reset}] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            f'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly! e.g. {p_norm_sq.max()}')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)  # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00  # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])  # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf  # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)  # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]  # [..., N_samples]
    denom = torch.where(denom < TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def render_single_image(rank, world_size, models, ray_sampler, chunk_size, iteration):
    ##### parallel rendering of a single image
    ray_sampler.update_rays(models['camera_mgr'])
    ray_batch = ray_sampler.get_all()

    fixed = 0
    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        fixed = world_size - (ray_batch['ray_d'].shape[0] % world_size)
        for p in ray_batch:
            if ray_batch[p] is not None:
                ray_batch[p] = torch.cat((ray_batch[p], ray_batch[p][-fixed:]), dim=0)
    #     raise Exception('Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(ray_batch['ray_d'].shape[0],
    #                                                                                                                                  world_size))

    # split into ranks; make sure different processes don't overlap
    rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                # bg_depth = torch.linspace(0., 1., N_samples).view(
                #     [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                # bg_weights = ret['bg_weights'].clone().detach()
                # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                # bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                # bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                #                               N_samples=N_samples, det=True)  # [..., N_samples]
                # bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                # del bg_weights
                # del bg_depth_mid
                # del bg_depth_samples
                torch.cuda.empty_cache()

            with autocast():
                with torch.no_grad():
                    ret = net(ray_o, ray_d, fg_far_depth, fg_depth, iteration, img_name=ray_sampler.img_path)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                sh = list(ret_merge_chunk[m][key].shape[1:])
                ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
                torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0)
                if fixed > 0:
                    ret_merge_rank[m][key] = ret_merge_rank[m][key][:-fixed]
                ret_merge_rank[m][key] = ret_merge_rank[m][key].reshape(
                    (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None


def log_view_to_tb(writer, global_step, log_data, gt_events, gt_rgb, mask, prefix=''):
    # rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    events_im = img_HWC2CHW(torch.from_numpy(gt_events))
    events_im = (events_im)/40+0.5
    writer.add_image(prefix + 'events_gt', events_im, global_step)

    rgb_im = img_HWC2CHW(torch.from_numpy(gt_rgb))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    for m in range(len(log_data)):
        # rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        # rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        writer.add_image(prefix + 'level_{}/rgb_norm'.format(m), (rgb_im-rgb_im.min(2, True)[0].min(1, True)[0])/(0.001+rgb_im.max(2,True)[0].max(1,True)[0]-rgb_im.min(2,True)[0].min(1,True)[0]), global_step)
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        writer.add_image(prefix + 'level_{}/fg_rgb_norm'.format(m), (rgb_im-rgb_im.min(2, True)[0].min(1, True)[0])/(0.001+rgb_im.max(2,True)[0].max(1,True)[0]-rgb_im.min(2,True)[0].min(1,True)[0]), global_step)
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)

        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)

        if 'fg_ldist' in log_data[m]:
            ldist = log_data[m]['fg_ldist']
            ldist_im = img_HWC2CHW(colorize(ldist, cmap_name='jet', append_cbar=True,
                                            mask=mask))
            writer.add_image(prefix + 'level_{}/fg_ldist'.format(m), ldist_im, global_step)

        # rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
        # rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        # writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)
        # depth = log_data[m]['bg_depth']
        # depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
        #                                 mask=mask))
        # writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
        bg_lambda = log_data[m]['bg_lambda']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
                                            mask=mask))
        writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)


def setup(rank, world_size):
    # initialize the process group
    slurmjob = os.environ.get('SLURM_JOB_ID', '')
    os.environ['MASTER_ADDR'] = 'localhost'
    if len(slurmjob) > 0:
        os.environ['MASTER_PORT'] = str(12000+int(slurmjob)%10000)
        logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on slurmjob ' + slurmjob)
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        try:
            os.environ['MASTER_PORT'] = '12413'
            logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on first try')
            torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        except RuntimeError:
            try:
                os.environ['MASTER_PORT'] = '12612'
                logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on second try')
                torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
            except RuntimeError:
                os.environ['MASTER_PORT'] = '15125'
                logger.info('using master port ' + os.environ['MASTER_PORT'] + ' based on third try')
                torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def create_nerf(rank, args, camera_mgr, load_camera_mgr=True):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777+args.seed_offset)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    img_names = None
    # load training image names for autoexposure
    f = os.path.join(args.basedir, args.expname, 'train_images.json')
    while not os.path.exists(f):
        time.sleep(5)
    time.sleep(5)
    with open(f) as file:
        img_names = json.load(file)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        # net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

    models['camera_mgr'] = camera_mgr.to(rank)

    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])

        if load_camera_mgr:
            name = 'camera_mgr'
            models[name].load_state_dict(to_load[name])

    return start, models


def ddp_train_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    elif torch.cuda.get_device_properties(rank).total_memory / 1e9 > 7:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096
    else:
        logger.info('setting batch size according to 4G gpu')
        args.N_rand = 512//4
        args.chunk_size = 4096//4

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
    torch.distributed.barrier()

    camera_mgr = CameraManager(learnable=False)
    ray_samplers = load_event_data_split(args.datadir, args.scene, camera_mgr=camera_mgr, split=args.train_split,
                                         skip=args.trainskip, max_winsize=args.winsize,
                                         use_ray_jitter=args.use_ray_jitter, is_colored=args.is_colored,
                                         polarity_offset=args.polarity_offset, cycle=args.is_cycled,
                                         is_rgb_only=args.is_rgb_only, randomize_winlen=args.use_random_window_len,
                                         win_constant_count=args.use_window_constant_count)
    val_ray_samplers = load_event_data_split(args.datadir, args.scene, camera_mgr=camera_mgr, split='validation',
                                         skip=args.testskip, max_winsize=1,
                                         use_ray_jitter=args.use_ray_jitter, is_colored=args.is_colored,
                                         polarity_offset=args.polarity_offset, cycle=args.is_cycled,
                                         is_rgb_only=args.is_rgb_only, randomize_winlen=args.use_random_window_len,
                                         win_constant_count=0)

    # write training image names for autoexposure
    if rank == 0:
        f = os.path.join(args.basedir, args.expname, 'train_images.json')
        with open(f, 'w') as file:
            img_names = [ray_samplers[i].img_path for i in range(len(ray_samplers))]
            json.dump(img_names, file, indent=2)

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args, camera_mgr)

    ##### important!!!
    # make sure different processes sample different rays
    np.random.seed((rank + 1) * 777+args.seed_offset)
    # make sure different processes have different perturbations in depth samples
    torch.manual_seed((rank + 1) * 777+args.seed_offset)

    ##### only main process should do the logging
    if rank == 0:
        writer = SummaryWriter(os.path.join(args.basedir, args.expname))

    scaler = GradScaler()
    # start training
    what_val_to_log = 0  # helper variable for parallel rendering of a image
    what_train_to_log = 0
    for global_step in range(start + 1, start + 1 + args.N_iters):
        time0 = time.time()
        scalars_to_log = OrderedDict()
        ### Start of core optimization loop
        scalars_to_log['resolution'] = ray_samplers[0].resolution_level
        # randomly sample rays and move to device
        i = np.random.randint(low=0, high=len(ray_samplers))
        ray_samplers[i].update_rays(models['camera_mgr'])

        # current_neg_ratio = min(global_step/args.neg_ratio_anneal, 1.0) * args.neg_ratio
        current_neg_ratio = args.neg_ratio if global_step > args.neg_ratio_anneal else 0.0
        scalars_to_log['neg_ratio'] = current_neg_ratio

        ray_batch = ray_samplers[i].random_sample(args.N_rand, center_crop=False, neg_ratio=current_neg_ratio)
        for key in ray_batch:
            if torch.is_tensor(ray_batch[key]):
                ray_batch[key] = ray_batch[key].to(rank)

        # forward and backward
        dots_sh = list(ray_batch['ray_d'].shape[:-1])  # number of rays
        # all_rets = []  # results on different cascade levels
        for m in range(models['cascade_level']):
            optim = models['optim_{}'.format(m)]
            net = models['net_{}'.format(m)]

            optim.zero_grad()
            with autocast():
                # sample depths
                N_samples = models['cascade_samples'][m]
                if m == 0:
                    # foreground depth
                    # prev_fg_far_depth = intersect_sphere(ray_batch['prev_ray_o'], ray_batch['prev_ray_d'])  # [...,]
                    # prev_fg_near_depth = ray_batch['min_depth']  # [..., ]
                    # prev_step = (prev_fg_far_depth - prev_fg_near_depth) / (N_samples - 1)
                    # prev_fg_depth = torch.stack([prev_fg_near_depth + i * prev_step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                    # prev_fg_depth = perturb_samples(prev_fg_depth)  # random perturbation during training

                    fg_far_depth = intersect_sphere(ray_batch['ray_o'], ray_batch['ray_d'])  # [...,]
                    fg_near_depth = ray_batch['min_depth']  # [..., ]
                    step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                    fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]
                    fg_depth = perturb_samples(fg_depth)  # random perturbation during training

                else:
                    # sample pdf and concat with earlier samples
                    # prev_fg_weights = prev_ret['fg_weights'].clone().detach()
                    # prev_fg_depth_mid = .5 * (prev_fg_depth[..., 1:] + prev_fg_depth[..., :-1])  # [..., N_samples-1]
                    # prev_fg_weights = prev_fg_weights[..., 1:-1]  # [..., N_samples-2]
                    # prev_fg_depth_samples = sample_pdf(bins=prev_fg_depth_mid, weights=prev_fg_weights,
                    #                                    N_samples=N_samples, det=False)  # [..., N_samples]
                    # prev_fg_depth, _ = torch.sort(torch.cat((prev_fg_depth, prev_fg_depth_samples), dim=-1))


                    # sample pdf and concat with earlier samples
                    fg_weights = ret['fg_weights'].clone().detach()
                    fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                    fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                    fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                                  N_samples=N_samples, det=False)  # [..., N_samples]
                    fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                    # sample pdf and concat with earlier samples
                    # bg_weights = ret['bg_weights'].clone().detach()
                    # bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                    # bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                    # bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                    #                               N_samples=N_samples, det=False)  # [..., N_samples]
                    # bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                if not args.is_rgb_only:
                    prev_ret = net(ray_batch['prev_ray_o'], ray_batch['prev_ray_d'], fg_far_depth, fg_depth, global_step,
                                   img_name=ray_batch['img_name'])

                ret = net(ray_batch['ray_o'], ray_batch['ray_d'], fg_far_depth, fg_depth, global_step,
                          img_name=ray_batch['img_name'])

                # all_rets.append(ret)

                events_gt = ray_batch['events'].to(rank)
                # rgb_gt = ray_batch['rgb'].to(rank)
                # prev_rgb_gt = ray_batch['prev_rgb'].to(rank)
                color_mask = ray_batch['color_mask'].to(rank)

                # cur_rgb_gt = ray_batch['rgb'].to(rank)
                #
                # rgb_loss = img2mse(ret['rgb'], rgb_gt)
                # loss = rgb_loss

                # diff = torch.log(ret['rgb']+1e-6)-torch.log(prev_ret['rgb']+1e-6)
                # event_mask = events_gt[..., 0] != 0
                event_mask = None
                # diff = ret['rgb']-prev_ret['rgb']
                if not args.is_rgb_only:
                    eps = args.tonemap_eps

                    def srgb_to_abs(x):
                        linmask = x <= 0.04045
                        linval = x/12.92
                        expval = ((x+0.055)/1.055)**2.4
                        res = linval*linmask+(~linmask)*expval
                        return res

                    diff = torch.log(ret['rgb']**2.2+eps)-torch.log(prev_ret['rgb']**2.2+eps)
                    # diff = torch.log(srgb_to_abs(ret['rgb'])+eps)-torch.log(srgb_to_abs(prev_ret['rgb'])+eps)
                    # diff = srgb_to_abs(ret['rgb'])-srgb_to_abs(prev_ret['rgb'])
                else:
                    diff = ret['rgb']*0
                diff = diff * color_mask
                events_gt = events_gt * color_mask
                # r = np.array([[1, 0], [0, 0]])
                # g = np.array([[0, 1], [1, 0]])
                # b = np.array([[0, 0], [0, 1]])
                # diff = ret['rgb']
                #
                # def corr(a, b):
                #     a = a.view(a.shape[0], -1)
                #     b = b.view(b.shape[0], -1)
                #     return ((a*b).mean(1)/torch.sqrt(((a**2).mean(1)+1e-6))/torch.sqrt(((b**2).mean(1)))).mean()
                # print(events_gt.min(), events_gt.max(), diff.min(), diff.max())
                # print((ray_batch['prev_ray_o']-ray_batch['ray_o']).std(),
                #     (ray_batch['prev_ray_d']-ray_batch['ray_d']).std())

                # event_loss = corr(diff, events_gt)
                # event_random_loss = corr(diff*0, events_gt)
                # THR = 0.1
                # THR = 0.5
                THR = args.event_threshold
                event_loss = img2mse(diff, events_gt*THR, event_mask)
                event_random_loss = img2mse(diff*0, events_gt*THR, event_mask)
                # #
                # prev_range_loss = torch.mean(torch.relu(0.1-prev_ret['fg_rgb'])+torch.relu(prev_ret['fg_rgb']-0.9))
                # curr_range_loss = torch.mean(torch.relu(0.1-ret['fg_rgb'])+torch.relu(ret['fg_rgb']-0.9))
                # range_loss = prev_range_loss+curr_range_loss
                #
                # lambda_loss = torch.mean(ret['bg_lambda']**2)+torch.mean(prev_ret['bg_lambda']**2)
                # #
                # loss = event_loss+0.1*range_loss+0.1*lambda_loss
                if args.is_rgb_only:
                    mask_gt = ray_batch['mask'].to(rank)
                    rgb_gt = ray_batch['rgb'].to(rank)

                    rgb_loss = img2mse(ret['rgb'], rgb_gt, mask=mask_gt)
                    event_loss = rgb_loss

                loss = event_loss
                # loss = rgb_loss
                # loss = img2mse(ret['rgb'], rgb_gt)
                # loss = (img2mse(prev_ret['rgb'], prev_rgb_gt)+img2mse(ret['rgb'], rgb_gt))/2
                # loss = (img2mse(prev_ret['rgb'], prev_rgb_gt, mask=events_gt[..., 0]!=0)+img2mse(ret['rgb'], rgb_gt, mask=events_gt[..., 0]!=0))/2
                # prev_scale = min(1, global_step/5000)
                # diff_gt = rgb_gt-prev_rgb_gt*prev_scale
                # loss = img2mse(ret['rgb'], (diff_gt-diff_gt.min())/(diff_gt.max()-diff_gt.min()))
                # loss = img2mse(ret['rgb']-prev_ret['rgb']*prev_scale, diff_gt)+0.1*range_loss

                if args.use_ldist_reg:
                    ldist = ret['fg_ldist'].mean()
                    loss = loss + ldist * args.ldist_reg

                scalars_to_log['level_{}/loss'.format(m)] = loss.item()
                scalars_to_log['level_{}/pnsr'.format(m)] = mse2psnr(event_loss.item())
                # scalars_to_log['level_{}/pnsr_rgb'.format(m)] = mse2psnr(rgb_loss.item())
                # scalars_to_log['level_{}/prev_scale'.format(m)] = prev_scale.item()
                # scalars_to_log['level_{}/range_loss'.format(m)] = range_loss.item()
                # scalars_to_log['level_{}/lambda_loss'.format(m)] = lambda_loss.item()
                scalars_to_log['level_{}/random_loss'.format(m)] = event_random_loss.item()
                # scalars_to_log['level_{}/real_div_random'.format(m)] = (event_loss.item()/max(0.01, event_random_loss.item()))
                # scalars_to_log['level_{}/real_minus_random'.format(m)] = (event_loss-event_random_loss).item()
                # scalars_to_log['level_{}/alpha_loss'.format(m)] = alpha_loss.item()
                if args.use_ldist_reg:
                    scalars_to_log['level_{}/ldist'.format(m)] = ldist.item()
            scaler.scale(loss).backward()
            # for pgi, pg in enumerate(optim.param_groups):
            #     for pi, p in enumerate(pg['params']):
            #         scalars_to_log['level_{}_grad_norm/{}_{}'.format(m, pgi, pi)] = torch.mean(p.grad**2)**0.5
            scaler.step(optim)

            # # clean unused memory
            # torch.cuda.empty_cache()

        scaler.update()
        ### end of core optimization loop
        dt = time.time() - time0
        scalars_to_log['iter_time'] = dt

        ### only main process should do the logging
        if rank == 0 and (global_step % args.i_print == 0 or global_step < 10):
            logstr = '{} step: {} '.format(args.expname, global_step)
            for k in scalars_to_log:
                logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                writer.add_scalar(k, scalars_to_log[k], global_step)
            logger.info(logstr)

        ### each process should do this; but only main process merges the results
        if global_step % args.i_img == 0 or global_step == start + 1:
            #### critical: make sure each process is working on the same random image
            time0 = time.time()
            idx = what_val_to_log % len(val_ray_samplers)
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                # with record_function("render_single_image_val"):
            log_data = render_single_image(rank, args.world_size, models, val_ray_samplers[idx], args.chunk_size,
                                                   global_step)
            # prof.export_chrome_trace("render.json")
            # prof.export_stacks("render_stacks_cpu", metric='self_cpu_time_total')
            # prof.export_stacks("render_stacks_cuda", metric='self_cuda_time_total')
            what_val_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random validation view in {} seconds'.format(dt))
                log_view_to_tb(writer, global_step, log_data,
                               gt_events=val_ray_samplers[idx].get_img(),
                               gt_rgb=val_ray_samplers[idx].get_rgb(),
                               mask=None,
                               prefix='val/')

            time0 = time.time()
            idx = what_train_to_log % len(ray_samplers)
            log_data = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size,
                                           global_step)
            what_train_to_log += 1
            dt = time.time() - time0
            if rank == 0:  # only main process should do this
                logger.info('Logged a random training view in {} seconds'.format(dt))
                log_view_to_tb(writer, global_step, log_data,
                               gt_events=ray_samplers[idx].get_img(),
                               gt_rgb=ray_samplers[idx].get_rgb(),
                               mask=None,
                               prefix='train/')
                writer.flush()

            del log_data
            torch.cuda.empty_cache()

        if rank == 0 and (global_step % args.i_weights == 0 and global_step > 0):
            # saving checkpoints and logging
            fpath = os.path.join(args.basedir, args.expname, 'model_{:06d}.pth'.format(global_step))
            to_save = OrderedDict()
            for m in range(models['cascade_level']):
                name = 'net_{}'.format(m)
                to_save[name] = models[name].state_dict()

                name = 'optim_{}'.format(m)
                to_save[name] = models[name].state_dict()

            name = 'camera_mgr'
            to_save[name] = models[name].state_dict()

            torch.save(to_save, fpath)

    # clean up for multi-processing
    cleanup()


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v == 'True':
            return True
        elif v == 'False':
            return False
        else:
            raise configargparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--slurmjob", type=str, default='', help='slurm job id')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')

    # ablation options
    parser.add_argument("--use_annealing", type=str2bool, default=True)
    parser.add_argument("--use_ray_jitter", type=str2bool, default=True)
    parser.add_argument("--use_ldist_reg", type=str2bool, default=True)
    parser.add_argument("--use_pe", type=str2bool, default=True)
    parser.add_argument("--use_random_window_len", type=str2bool, default=True)
    parser.add_argument("--use_window_constant_count", type=int, default=0)

    parser.add_argument("--seed_offset", type=int, default=0, help='random seed offset')

    # dataset options
    parser.add_argument("--datadir", type=str, default=None, help='input data directory')
    parser.add_argument("--scene", type=str, default=None, help='scene name')
    parser.add_argument("--train_split", type=str, default='train', help='training split')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    parser.add_argument("--trainskip", type=int, default=1,
                        help='will load 1/N images from train sets, useful for large datasets like deepvoxels')

    parser.add_argument("--winsize", type=int, default=1,
                        help='event window size')

    parser.add_argument("--is_colored", type=str2bool, default=False,
                        help='are events from the color event camera')

    parser.add_argument("--is_cycled", type=str2bool, default=False,
                        help='are events cycled')

    parser.add_argument("--is_rgb_only", type=str2bool, default=False,
                        help='use regular nerf rgb loss and disregard events completely')

    parser.add_argument("--event_threshold", type=float, default=0.5, help='event threshold')
    parser.add_argument("--polarity_offset", type=float, default=0.0, help='polarity offset')

    parser.add_argument("--tonemap_eps", type=float, default=1e-5, help='tonemapping eps')

    parser.add_argument("--bg_color", type=float, default=159., help='background color in srgb')

    # model size
    parser.add_argument("--netdepth", type=int, default=8, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--activation", type=str, default=None, help='activation function (relu, elu, sine, garf, tanh)')
    parser.add_argument("--garf_sigma", type=float, default=1.0, help='garf activation function sigma')

    parser.add_argument("--with_bg", type=str2bool, default=False, help='use background network')

    parser.add_argument("--crop_y_min", type=float, default=-1, help='zero density of everything below')
    parser.add_argument("--crop_y_max", type=float, default=1, help='zero density of everything above')
    parser.add_argument("--crop_r", type=float, default=1, help='zero density of everything outside of x2+z2<=r')


    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # batch size
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 2,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 8,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')

    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')

    # cascade training
    parser.add_argument("--cascade_level", type=int, default=2,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')

    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')

    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true',
                        help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    parser.add_argument("--ldist_reg", type=float, default=0.001)

    parser.add_argument("--neg_ratio", type=float, default=0,
                        help='ratio of samples at pixels without events')

    parser.add_argument("--neg_ratio_anneal", type=int, default=0,
                        help='number of negative ratio anneal iterations')

    parser.add_argument("--init_gain", type=float, default=5,
                        help='initialisation gain')


    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')

    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--N_anneal", type=int, default=100000,
                        help='number of embedder anneal iterations')
    parser.add_argument("--N_anneal_min_freq", type=int, default=0,
                        help='number of embedder frequencies to start annealing from')
    parser.add_argument("--N_anneal_min_freq_viewdirs", type=int, default=0,
                        help='number of viewdir embedder frequencies to start annealing from')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    if 'SLURM_JOB_ID' in os.environ:
        args.slurmjob = os.environ['SLURM_JOB_ID']
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    train()
